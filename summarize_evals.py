import json
import glob
import os

files = glob.glob("evaluation_output/*/*/evaluation_results.json")

results = []
for f in files:
    with open(f, 'r') as file:
        data = json.load(file)
        parts = f.split(os.sep)
        run_name = parts[1]
        model_type = parts[2]
        
        results.append({
            "run": run_name,
            "type": model_type,
            "mean_iou": data.get("mean_iou", 0),
            "class_iou": data.get("class_iou", {})
        })

# Sort by Mean IoU descending
results.sort(key=lambda x: x["mean_iou"], reverse=True)

markdown_content = "# Evaluation Results Comparison\n\n"

markdown_content += "| Run Name | Model Type | Mean IoU | Sea Surface IoU | Oil Spill IoU | Look Alike IoU | Ship IoU | Land IoU |\n"
markdown_content += "|----------|------------|----------|-----------------|---------------|----------------|----------|----------|\n"

for r in results:
    markdown_content += f"| {r['run']} | {r['type']} | {r['mean_iou']:.4f} | {r['class_iou'].get('sea_surface', 0):.4f} | {r['class_iou'].get('oil_spill', 0):.4f} | {r['class_iou'].get('look_alike', 0):.4f} | {r['class_iou'].get('ship', 0):.4f} | {r['class_iou'].get('land', 0):.4f} |\n"

with open("evaluation_comparison.md", "w") as markdown_file:
    markdown_file.write(markdown_content)

print("Created evaluation_comparison.md")
