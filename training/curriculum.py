import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.curriculum_cfg = config.get("curriculum", {})
        logger.warning(
            "training.curriculum.CurriculumScheduler is deprecated. "
            "Use data.dataloader.CurriculumDataLoaderFactory for active sampling behavior."
        )

        self.phase_ranges = self._parse_phase_ranges()
        self.phase_numbers = [phase for phase, _, _ in self.phase_ranges]

        # Backward-compatible fields retained for legacy callers.
        self.phase1_end = self._get_phase_end(1, default=50)
        self.phase2_end = self._get_phase_end(2, default=150)
        self.phase3_end = self._get_phase_end(3, default=300)

        self.stagger_transitions = self._parse_stagger_transitions()
        self.stagger_1_to_2 = self._get_transition_config(1, 2)
        self.stagger_2_to_3 = self._get_transition_config(2, 3)

        self._sampling_transition_epochs = self._collect_transition_epochs(
            "sampling_epoch"
        )
        self._resize_transition_epochs = self._collect_transition_epochs("resize_epoch")
        self._gamma_transition_epochs = self._collect_transition_epochs("gamma_epoch")
        self._weights_transition_epochs = self._collect_transition_epochs(
            "weights_epoch"
        )

        self._current_phase = None
        self._applied_transitions: set = set()

        if self.stagger_transitions:
            transition_parts = []
            for from_phase, to_phase, trans_cfg in self.stagger_transitions:
                transition_parts.append(
                    f"{from_phase}->{to_phase} "
                    f"sampling@{trans_cfg.get('sampling_epoch', '-')}, "
                    f"resize@{trans_cfg.get('resize_epoch', '-')}, "
                    f"gamma@{trans_cfg.get('gamma_epoch', '-')}, "
                    f"weights@{trans_cfg.get('weights_epoch', '-')}"
                )
            logger.info("CurriculumScheduler: " + "; ".join(transition_parts))
        else:
            logger.info(
                "CurriculumScheduler: no stagger transitions configured; "
                "using direct phase epoch ranges %s",
                self.phase_ranges,
            )

    def _parse_phase_ranges(self) -> list[tuple[int, int, int]]:
        phase_ranges: list[tuple[int, int, int]] = []
        phase_key_re = re.compile(r"^phase_(\d+)$")
        for key, phase_cfg in self.curriculum_cfg.items():
            if not isinstance(key, str):
                continue
            match = phase_key_re.match(key)
            if match is None or not isinstance(phase_cfg, dict):
                continue
            epochs = phase_cfg.get("epochs")
            if not isinstance(epochs, (list, tuple)) or len(epochs) < 2:
                continue
            phase = int(match.group(1))
            phase_ranges.append((phase, int(epochs[0]), int(epochs[1])))

        if phase_ranges:
            phase_ranges.sort(key=lambda item: item[0])
            return phase_ranges

        # Legacy fallback when curriculum phases are not explicitly declared.
        return [(1, 1, 50), (2, 51, 150), (3, 151, 300)]

    def _parse_stagger_transitions(self) -> list[tuple[int, int, dict]]:
        transitions: list[tuple[int, int, dict]] = []
        trans_key_re = re.compile(r"^phase_(\d+)_to_(\d+)$")
        for key, value in self.curriculum_cfg.items():
            if not isinstance(key, str):
                continue
            match = trans_key_re.match(key)
            if match is None or not isinstance(value, dict):
                continue
            from_phase = int(match.group(1))
            to_phase = int(match.group(2))
            transitions.append((from_phase, to_phase, value))

        transitions.sort(key=lambda item: item[1])
        return transitions

    def _collect_transition_epochs(self, field: str) -> set[int]:
        epochs: set[int] = set()
        for _, _, trans_cfg in self.stagger_transitions:
            if field in trans_cfg:
                epochs.add(int(trans_cfg[field]))
        return epochs

    def _get_phase_end(self, phase: int, default: int) -> int:
        phase_cfg = self.curriculum_cfg.get(f"phase_{phase}", {})
        epochs = phase_cfg.get("epochs", [])
        if isinstance(epochs, (list, tuple)) and len(epochs) >= 2:
            return int(epochs[1])
        return int(default)

    def _get_transition_config(self, from_phase: int, to_phase: int) -> dict:
        return self.curriculum_cfg.get(f"phase_{from_phase}_to_{to_phase}", {})

    def _phase_from_ranges(self, epoch: int) -> int:
        for phase, start_epoch, end_epoch in self.phase_ranges:
            if start_epoch <= epoch <= end_epoch:
                return phase
        if epoch < self.phase_ranges[0][1]:
            return self.phase_ranges[0][0]
        return self.phase_ranges[-1][0]

    def _phase_from_stagger(self, epoch: int, epoch_field: str) -> int:
        if not self.stagger_transitions:
            return self._phase_from_ranges(epoch)

        current_phase = self.phase_ranges[0][0]
        sorted_transitions = sorted(
            self.stagger_transitions,
            key=lambda item: int(item[2].get(epoch_field, 10**9)),
        )
        for _, to_phase, trans_cfg in sorted_transitions:
            if epoch_field not in trans_cfg:
                continue
            transition_epoch = int(trans_cfg[epoch_field])
            if epoch >= transition_epoch:
                current_phase = to_phase
            else:
                break
        return current_phase

    def get_phase(self, epoch: int) -> int:
        return self.get_staggered_sampling_phase(epoch)

    def should_advance_to_phase2(
        self,
        current_epoch: int,
        current_metrics: dict,
    ) -> bool:
        if 2 not in self.phase_numbers:
            return False

        force_epoch = int(self.curriculum_cfg.get("phase2_force_epoch", 70))
        min_oil_iou = float(self.curriculum_cfg.get("phase2_min_oil_iou", 0.25))

        if current_epoch < (self.phase1_end + 1):
            return False
        if current_epoch >= force_epoch:
            return True
        return current_metrics.get("oil_spill_iou", 0.0) >= min_oil_iou

    def is_sampling_transition(self, epoch: int) -> bool:
        return epoch in self._sampling_transition_epochs

    def is_resize_transition(self, epoch: int) -> bool:
        return epoch in self._resize_transition_epochs

    def is_gamma_transition(self, epoch: int) -> bool:
        return epoch in self._gamma_transition_epochs

    def is_weights_transition(self, epoch: int) -> bool:
        return epoch in self._weights_transition_epochs

    def get_staggered_focal_gamma(self, epoch: int) -> float:
        phase = self._phase_from_stagger(epoch, "gamma_epoch")
        default_gamma = 2.5 if phase == 2 else 2.0
        return float(
            self.curriculum_cfg.get(f"phase_{phase}", {}).get(
                "focal_gamma", default_gamma
            )
        )

    def get_staggered_sampling_phase(self, epoch: int) -> int:
        return self._phase_from_stagger(epoch, "sampling_epoch")

    def get_staggered_weights_phase(self, epoch: int) -> int:
        return self._phase_from_stagger(epoch, "weights_epoch")

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        phase = self.get_staggered_weights_phase(epoch)
        base_cfg = self.config.get("loss", {})

        focal_w = float(base_cfg.get("focal", {}).get("weight", 0.4))
        dice_w = float(base_cfg.get("dice", {}).get("weight", 0.3))
        boundary_cfg = base_cfg.get("boundary", {})
        boundary_w = float(
            boundary_cfg.get(
                f"weight_phase{phase}",
                boundary_cfg.get("weight", 0.2),
            )
        )
        confusion_cfg = base_cfg.get("confusion_penalty", {})
        confusion_w = float(
            confusion_cfg.get(
                f"weight_phase{phase}",
                confusion_cfg.get("weight", 0.1),
            )
        )
        if phase == 2:
            confusion_w = float(confusion_cfg.get("phase2_weight", confusion_w))
        if phase == 3:
            confusion_w = float(confusion_cfg.get("phase3_weight", confusion_w))

        return {
            "focal": focal_w,
            "dice": dice_w,
            "boundary": boundary_w,
            "confusion_penalty": confusion_w,
            "contrastive": self.get_contrastive_weight(epoch),
        }

    def get_contrastive_weight(self, epoch: int) -> float:
        if epoch <= 1:
            return 0.05
        if epoch <= 75:
            return 0.05 + (0.10 * (epoch - 1) / 74.0)
        return 0.15

    def get_sampling_config(self, epoch: int) -> Dict[str, Any]:
        phase = self.get_phase(epoch)
        phase_key = f"phase_{phase}"
        phase_cfg = self.curriculum_cfg.get(phase_key, {})

        return {
            "phase": phase,
            "sampling": phase_cfg.get("sampling", "standard"),
            "ship_oversample_factor": phase_cfg.get("ship_oversample_factor", 1.0),
        }

    def log_phase_transition(self, epoch: int) -> None:
        phase = self.get_phase(epoch)
        phase_cfg = self.curriculum_cfg.get(f"phase_{phase}", {})
        desc = (
            f"Configured phase settings "
            f"(ship={phase_cfg.get('ship_oversample_factor', 1.0)}x, "
            f"oil={phase_cfg.get('oil_spill_oversample_factor', 1.0)}x, "
            f"look_alike={phase_cfg.get('look_alike_oversample_factor', 1.0)}x)"
        )
        logger.info(f"{'=' * 60}")
        logger.info(f"CURRICULUM TRANSITION: Entering Phase {phase} at epoch {epoch}")
        logger.info(f"  Strategy: {desc}")
        logger.info(f"  Focal gamma: {self.get_staggered_focal_gamma(epoch)}")
        s = None
        for _, to_phase, trans_cfg in self.stagger_transitions:
            if to_phase == phase:
                s = trans_cfg
                break
        if s:
            logger.info(
                f"  Stagger: sampling@{s['sampling_epoch']}, "
                f"resize@{s['resize_epoch']}, "
                f"gamma@{s['gamma_epoch']}, "
                f"weights@{s['weights_epoch']}"
            )
        logger.info(f"{'=' * 60}")
        self._current_phase = phase

    def log_stagger_event(self, epoch: int, change_type: str) -> None:
        logger.info(f"  STAGGER [{change_type}] applied at epoch {epoch}")

    def mark_transition_applied(self, transition_id: str) -> bool:
        if transition_id in self._applied_transitions:
            logger.warning(
                f"[Curriculum] Transition '{transition_id}' already applied — "
                f"skipping (checkpoint resume guard)"
            )
            return False
        self._applied_transitions.add(transition_id)
        logger.info(f"[Curriculum] Transition applied: {transition_id}")
        return True

    def get_applied_transitions(self) -> list:
        return sorted(self._applied_transitions)

    def restore_applied_transitions(self, transitions: list) -> None:
        self._applied_transitions = set(transitions)
