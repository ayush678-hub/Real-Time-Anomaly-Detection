import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from feature_extractor import extract_frames, CLASSES, FOLDER_TO_CLASS
from model import VideoClassifier

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
CHECKPOINT   = os.path.join(os.path.dirname(__file__), "best_model.pth")
BATCH_SIZE   = 4
EPOCHS       = 20
LR           = 1e-4
VAL_SPLIT    = 0.15
# num_workers=0 is required on Windows to avoid multiprocessing spawn errors
NUM_WORKERS  = 0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


class LazyVideoDataset(Dataset):
    """Reads frames from disk on-demand — avoids loading all videos into RAM."""

    def __init__(self, samples):
        # samples: list of (video_path, class_idx)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        frames = extract_frames(path)
        if frames is None:
            frames = np.zeros((16, 3, 112, 112), dtype=np.float32)
        return torch.tensor(frames, dtype=torch.float32), label


def collect_samples(dataset_dir):
    samples, labels = [], []
    for folder, class_name in FOLDER_TO_CLASS.items():
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        class_idx = CLASSES.index(class_name)
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".mp4"):
                samples.append((os.path.join(folder_path, fname), class_idx))
                labels.append(class_idx)
    return samples, labels


def train():
    print(f"Device: {DEVICE}")
    print("Scanning dataset...")
    samples, labels = collect_samples(DATASET_DIR)
    print(f"Total samples: {len(samples)}")

    dataset  = LazyVideoDataset(samples)
    val_size  = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Class-balanced loss weights
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=np.array(labels))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    model     = VideoClassifier(num_classes=len(CLASSES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()

        scheduler.step()
        train_acc = correct / train_size

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_correct += (model(xb).argmax(1) == yb).sum().item()
        val_acc = val_correct / val_size

        print(f"  loss={total_loss/train_size:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  [SAVED] Best model  (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.3f}")
    print(f"Checkpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    train()
