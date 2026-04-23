import cv2
import numpy as np
import os

CLASSES = [
    "abuse", "arrest", "arson", "assault", "burglary",
    "explosion", "fighting", "normal", "road_accident",
    "robbery", "shooting", "shoplifting", "stealing", "vandalism"
]

SEVERITY_MAP = {
    "normal": 1,
    "shoplifting": 3, "stealing": 4, "burglary": 5, "vandalism": 4,
    "arrest": 5, "abuse": 6, "assault": 7, "robbery": 7,
    "fighting": 7, "road_accident": 8, "arson": 9,
    "shooting": 9, "explosion": 10
}

ANOMALY_MAP = {c: "Normal" if c == "normal" else "Anomalous" for c in CLASSES}

FOLDER_TO_CLASS = {
    "abuse": "abuse", "arrest": "arrest", "arson": "arson",
    "assault": "assault", "burglary": "burglary", "explosion": "explosion",
    "fighting": "fighting", "normal": "normal",
    "road accidents": "road_accident", "robbery": "robbery",
    "shooting": "shooting", "shop lifting": "shoplifting",
    "stealing": "stealing", "vandalism": "vandalism"
}

NUM_FRAMES = 16
IMG_SIZE = 112


def extract_frames(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    frames = np.array(frames, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frames = (frames - mean) / std
    # (T, H, W, C) -> (T, C, H, W)
    return frames.transpose(0, 3, 1, 2)


def build_dataset(dataset_dir):
    X, y = [], []
    for folder, label in FOLDER_TO_CLASS.items():
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        class_idx = CLASSES.index(label)
        for fname in os.listdir(folder_path):
            if not fname.endswith(".mp4"):
                continue
            frames = extract_frames(os.path.join(folder_path, fname))
            if frames is not None:
                X.append(frames)
                y.append(class_idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
