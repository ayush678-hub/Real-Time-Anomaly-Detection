import os
import torch
from model import VideoClassifier
from feature_extractor import FOLDER_TO_CLASS, CLASSES

# --- Model test ---
model = VideoClassifier(num_classes=14).cuda()
x = torch.randn(2, 16, 3, 112, 112).cuda()
out = model(x)
print("Model forward pass OK")
print("Output shape:", out.shape)
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")

# --- Dataset scan ---
dataset_dir = os.path.join(os.path.dirname(__file__), "Dataset")
count = 0
class_counts = {}
for folder, class_name in FOLDER_TO_CLASS.items():
    folder_path = os.path.join(dataset_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"  [MISSING] {folder_path}")
        continue
    n = len([f for f in os.listdir(folder_path) if f.endswith(".mp4")])
    class_counts[class_name] = n
    count += n

print(f"\nDataset scan OK — Total videos: {count}")
for cls, n in sorted(class_counts.items()):
    print(f"  {cls:<20} {n} videos")
