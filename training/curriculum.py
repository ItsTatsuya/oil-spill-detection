import logging
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

        self.phase1_end = self.curriculum_cfg.get("phase_1", {}).get("epochs", [1, 50])[
            1
        ]
        self.phase2_end = self.curriculum_cfg.get("phase_2", {}).get(
            "epochs", [51, 150]
        )[1]
        self.phase3_end = self.curriculum_cfg.get("phase_3", {}).get(
            "epochs", [151, 300]
        )[1]

        self.stagger_1_to_2 = self.curriculum_cfg["phase_1_to_2"]
        self.stagger_2_to_3 = self.curriculum_cfg["phase_2_to_3"]
        self._current_phase = None
        self._applied_transitions: set = set()

        logger.info(
            f"CurriculumScheduler: "
            f"Phase 1->2 sampling@{self.stagger_1_to_2['sampling_epoch']}, "
            f"resize@{self.stagger_1_to_2['resize_epoch']}, "
            f"gamma@{self.stagger_1_to_2['gamma_epoch']}, "
            f"weights@{self.stagger_1_to_2['weights_epoch']}; "
            f"Phase 2->3 sampling@{self.stagger_2_to_3['sampling_epoch']}, "
            f"resize@{self.stagger_2_to_3['resize_epoch']}, "
            f"gamma@{self.stagger_2_to_3['gamma_epoch']}, "
            f"weights@{self.stagger_2_to_3['weights_epoch']}"
        )

    def get_phase(self, epoch: int) -> int:
        return self.get_staggered_sampling_phase(epoch)

    def should_advance_to_phase2(
        self,
        current_epoch: int,
        current_metrics: dict,
    ) -> bool:
        force_epoch = int(self.curriculum_cfg.get("phase2_force_epoch", 70))
        min_oil_iou = float(self.curriculum_cfg.get("phase2_min_oil_iou", 0.25))

        if current_epoch < (self.phase1_end + 1):
            return False
        if current_epoch >= force_epoch:
            return True
        return current_metrics.get("oil_spill_iou", 0.0) >= min_oil_iou


    def is_sampling_transition(self, epoch: int) -> bool:
        return (
            epoch == self.stagger_1_to_2["sampling_epoch"]
            or epoch == self.stagger_2_to_3["sampling_epoch"]
        )

    def is_resize_transition(self, epoch: int) -> bool:
        return (
            epoch == self.stagger_1_to_2["resize_epoch"]
            or epoch == self.stagger_2_to_3["resize_epoch"]
        )

    def is_gamma_transition(self, epoch: int) -> bool:
        return (
            epoch == self.stagger_1_to_2["gamma_epoch"]
            or epoch == self.stagger_2_to_3["gamma_epoch"]
        )

    def is_weights_transition(self, epoch: int) -> bool:
        return (
            epoch == self.stagger_1_to_2["weights_epoch"]
            or epoch == self.stagger_2_to_3["weights_epoch"]
        )

    def get_staggered_focal_gamma(self, epoch: int) -> float:
        gamma_1_to_2 = self.stagger_1_to_2["gamma_epoch"]
        gamma_2_to_3 = self.stagger_2_to_3["gamma_epoch"]
        if epoch < gamma_1_to_2:
            return float(self.curriculum_cfg.get("phase_1", {}).get("focal_gamma", 2.0))
        if epoch < gamma_2_to_3:
            return float(self.curriculum_cfg.get("phase_2", {}).get("focal_gamma", 2.5))
        return float(self.curriculum_cfg.get("phase_3", {}).get("focal_gamma", 2.0))

    def get_staggered_sampling_phase(self, epoch: int) -> int:
        s_1_to_2 = self.stagger_1_to_2["sampling_epoch"]
        s_2_to_3 = self.stagger_2_to_3["sampling_epoch"]
        if epoch < s_1_to_2:
            return 1
        if epoch < s_2_to_3:
            return 2
        return 3

    def get_staggered_weights_phase(self, epoch: int) -> int:
        w_1_to_2 = self.stagger_1_to_2["weights_epoch"]
        w_2_to_3 = self.stagger_2_to_3["weights_epoch"]
        if epoch < w_1_to_2:
            return 1
        if epoch < w_2_to_3:
            return 2
        return 3

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
        confusion_w = float(confusion_cfg.get("weight", 0.1))
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
        p1_ship = self.curriculum_cfg.get("phase_1", {}).get("ship_oversample_factor", 2.0)
        p2_ship = self.curriculum_cfg.get("phase_2", {}).get("ship_oversample_factor", 3.0)
        p3_ship = self.curriculum_cfg.get("phase_3", {}).get("ship_oversample_factor", 2.0)
        descriptions = {
            1: f"Standard sampling with inverse-frequency class weights (ship {p1_ship}x)",
            2: f"Ship-enriched sampling ({p2_ship}x oversampling) + increased focal gamma",
            3: f"Standard sampling (ship {p3_ship}x) + stronger confusion penalty",
        }
        desc = descriptions.get(phase, "Unknown phase")
        logger.info(f"{'=' * 60}")
        logger.info(f"CURRICULUM TRANSITION: Entering Phase {phase} at epoch {epoch}")
        logger.info(f"  Strategy: {desc}")
        logger.info(f"  Focal gamma: {self.get_staggered_focal_gamma(epoch)}")
        if phase == 2:
            s = self.stagger_1_to_2
        elif phase == 3:
            s = self.stagger_2_to_3
        else:
            s = None
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
