#  AI-Powered Video Surveillance System

An AI/ML-powered video surveillance system that analyzes CCTV footage to detect and classify unusual events such as violence, theft, and accidents. The system distinguishes normal behaviors from anomalies, assigns a severity score (1–10), and provides natural language explanations for each detection.

---

##  Problem Statement

Design an AI/ML-powered video surveillance system that analyzes real-time CCTV footage to detect and classify unusual events such as violence, theft, or accidents. The system must distinguish normal behaviors from anomalies despite challenges like limited anomaly data, noisy frames, and incomplete labels, while providing reliable, interpretable, and scalable alerts.

For example, in a crowded metro station during rush hour, the model should filter routine passenger movement but quickly flag a sudden fight or accident, explain why it was detected, and notify security staff in real time. Each detected event should be classified on a scale of **1–10**, where 1 represents normal activity and 10 indicates the most severe anomaly, enabling security teams to prioritize responses effectively.

---

##  Project Structure

```
AI FOR SOCIAL GOOD 1/
├── Dataset/
│   ├── abuse/
│   ├── arrest/
│   ├── arson/
│   ├── assault/
│   ├── burglary/
│   ├── explosion/
│   ├── fighting/
│   ├── normal/
│   ├── road accidents/
│   ├── robbery/
│   ├── shooting/
│   ├── shop lifting/
│   ├── stealing/
│   └── vandalism/
├── model.py
├── feature_extractor.py
├── train.py
├── predict.py
├── app.py
├── model_test.py
├── best_model.pth
├── train_log.txt
├── requirements.txt
└── PROBLEM STATEMENT.txt
```

---

##  Model Architecture

The model is a **CNN + LSTM hybrid** (`VideoClassifier` in `model.py`):

| Component | Details |
|---|---|
| CNN Backbone | MobileNetV2 (pretrained on ImageNet) |
| Feature Dim | 1280-dim vector per frame |
| Pooling | AdaptiveAvgPool2d → collapses spatial dims |
| Temporal Model | LSTM (hidden=256, 1 layer) |
| Classifier | Dropout → Linear → 14-class output |
| Input Shape | `(Batch, 16 frames, 3 channels, 112×112)` |

The CNN extracts spatial features from each frame independently, then the LSTM models the temporal sequence across all 16 frames to make a final classification.

---

##  Source Files

### `model.py`
Defines the `VideoClassifier` neural network — a CNN + LSTM hybrid:
- CNN backbone: MobileNetV2 (pretrained) extracts spatial features per frame → outputs 1280-dim feature vectors
- AdaptiveAvgPool2d collapses spatial dims to a single vector per frame
- LSTM processes the temporal sequence of frame features (hidden size=256, 1 layer)
- Final classifier: Dropout → Linear → 14-class output
- Input shape: `(Batch, 16 frames, 3 channels, 112×112 pixels)`

**Key classes & functions:**

| Name | Type | Description |
|---|---|---|
| `VideoClassifier` | Class | Main model — CNN + LSTM hybrid for video classification |
| `__init__()` | Method | Builds MobileNetV2 backbone, LSTM, and classifier head |
| `forward(x)` | Method | Reshapes input, extracts CNN features per frame, runs LSTM, returns logits |

---

### `feature_extractor.py`
Handles all data definitions and video preprocessing:
- `CLASSES` — list of 14 activity classes
- `SEVERITY_MAP` — maps each class to a severity score (1–10)
- `ANOMALY_MAP` — labels each class as "Normal" or "Anomalous"
- `FOLDER_TO_CLASS` — maps dataset folder names to class names (e.g. `"road accidents"` → `"road_accident"`)
- `extract_frames()` — reads a video with OpenCV, uniformly samples 16 frames, resizes to 112×112, normalizes with ImageNet mean/std, returns shape `(16, 3, 112, 112)`
- `build_dataset()` — loads all videos into memory as numpy arrays (for small-scale testing)

**Key constants & functions:**

| Name | Type | Description |
|---|---|---|
| `CLASSES` | List | 14 class labels in index order used by the model |
| `SEVERITY_MAP` | Dict | Class → severity score (1–10) |
| `ANOMALY_MAP` | Dict | Class → "Normal" or "Anomalous" |
| `FOLDER_TO_CLASS` | Dict | Dataset folder name → normalized class name |
| `NUM_FRAMES` | Constant | 16 — number of frames sampled per video |
| `IMG_SIZE` | Constant | 112 — frame resize resolution (pixels) |
| `extract_frames()` | Function | Reads video, samples 16 frames uniformly, resizes, normalizes, returns `(16, 3, 112, 112)` |
| `build_dataset()` | Function | Iterates all folders, calls `extract_frames()`, returns `(X, y)` numpy arrays |

---

### `train.py`
Full training pipeline:
- `LazyVideoDataset` — PyTorch Dataset that reads frames on-demand (avoids loading all videos into RAM)
- `collect_samples()` — scans dataset folders and builds a list of `(video_path, class_idx)` pairs
- Uses `compute_class_weight("balanced")` from sklearn to handle class imbalance
- Saves best model checkpoint to `best_model.pth` based on validation accuracy
- `NUM_WORKERS=0` required on Windows to avoid multiprocessing issues

**Training Config:**

| Parameter | Value |
|---|---|
| Batch Size | 4 |
| Epochs | 20 |
| Learning Rate | 1e-4 |
| Validation Split | 15% |
| Optimizer | Adam |
| Scheduler | StepLR (halves LR every 7 epochs) |
| Loss | CrossEntropyLoss (class-weighted) |

**Key classes & functions:**

| Name | Type | Description |
|---|---|---|
| `LazyVideoDataset` | Class | PyTorch Dataset — loads frames from disk on-demand per `__getitem__` call |
| `collect_samples()` | Function | Scans all dataset folders, returns `(samples, labels)` lists |
| `train()` | Function | Full training loop — data loading, loss, backprop, validation, checkpointing |

---

### `predict.py`
Inference module:
- `load_model()` — loads `VideoClassifier` and restores weights from `best_model.pth`
- `predict(video_path)` — extracts frames → runs model → applies softmax → returns a result dict with:
  - Predicted class
  - Anomaly status (Normal / Anomalous)
  - Severity score (1–10)
  - Confidence %
  - Natural language explanation
  - Key visual indicators
- `EXPLANATIONS` and `KEY_INDICATORS` — hardcoded per-class descriptions for interpretability
- Can be run standalone: `python predict.py <video.mp4>`

**Key constants & functions:**

| Name | Type | Description |
|---|---|---|
| `EXPLANATIONS` | Dict | Per-class natural language description of what was detected |
| `KEY_INDICATORS` | Dict | Per-class list of 3 visual cues the model looks for |
| `load_model()` | Function | Instantiates `VideoClassifier`, loads `best_model.pth` weights, sets eval mode |
| `predict()` | Function | End-to-end inference — frames → model → softmax → structured result dict |

---

### `app.py`
Gradio web UI:
- Loads model once at startup
- `analyze()` — calls `predict()` and formats results as a markdown table + raw JSON
- `severity_icon()` — maps score to severity icons: 🟢 Normal / 🟡 Mild / 🟠 Serious / 🔴 Critical
- UI: video upload → Analyze button → markdown summary + collapsible JSON output
- Runs locally with `demo.launch(inbrowser=True)`

**Key functions & UI components:**

| Name | Type | Description |
|---|---|---|
| `severity_icon()` | Function | Maps severity score to a colored emoji label |
| `analyze()` | Function | Calls `predict()`, formats markdown summary and JSON output |
| `video_input` | Gradio Component | Video upload widget |
| `analyze_btn` | Gradio Component | Primary button that triggers `analyze()` |
| `summary_out` | Gradio Component | Markdown display for detection results |
| `json_out` | Gradio Component | Collapsible JSON code block for raw output |

---

### `model_test.py`
Sanity check script:
- Verifies model forward pass with a random tensor `(2, 16, 3, 112, 112)` on CUDA
- Prints output shape and total parameter count
- Scans the Dataset folder and reports video counts per class

**What it checks:**

| Check | Description |
|---|---|
| Forward pass | Runs a random batch through the model, confirms output shape is `(2, 14)` |
| Parameter count | Prints total number of trainable parameters |
| Dataset scan | Lists each class folder and counts `.mp4` files present |

---

##  Dataset

~964 `.mp4` videos across 14 classes, sourced from the **UCF-Crime dataset**:

| Class | Folder | Severity | ~Videos |
|---|---|---|---|
| Normal | `normal/` | 1 | 60 |
| Shoplifting | `shop lifting/` | 3 | 50 |
| Stealing | `stealing/` | 4 | 114 |
| Vandalism | `vandalism/` | 4 | 50 |
| Arrest | `arrest/` | 5 | 50 |
| Burglary | `burglary/` | 5 | 100 |
| Abuse | `abuse/` | 6 | 50 |
| Assault | `assault/` | 7 | 51 |
| Fighting | `fighting/` | 7 | 50 |
| Robbery | `robbery/` | 7 | 150 |
| Road Accident | `road accidents/` | 8 | 90 |
| Arson | `arson/` | 9 | 49 |
| Shooting | `shooting/` | 9 | 50 |
| Explosion | `explosion/` | 10 | 50 |

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch |
| CNN Backbone | MobileNetV2 (torchvision, pretrained ImageNet) |
| Temporal Modeling | LSTM |
| Video Processing | OpenCV |
| Class Imbalance | scikit-learn `compute_class_weight` |
| Web UI | Gradio |
| Training Progress | tqdm |

---

##  Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

Checkpoint saved to `best_model.pth` after each best validation accuracy.

### 3. Run Inference on a Video

```bash
python predict.py path/to/video.mp4
```

### 4. Launch the Web UI

```bash
python app.py
```

Opens a Gradio interface in your browser at `http://localhost:7860`.

### 5. Run Model Sanity Check

```bash
python model_test.py
```

---

## Output Format

```json
{
  "predicted_class": "robbery",
  "anomaly": "Anomalous",
  "severity_score": 7,
  "confidence": 91.4,
  "explanation": "An individual is forcibly taking property from another person using threats or physical force.",
  "key_indicators": [
    "confrontational approach",
    "victim compliance under duress",
    "property forcibly taken"
  ]
}
```

---

##  Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
gradio>=4.0.0
httpx>=0.24.0,<0.28.0
tqdm>=4.65.0
Pillow>=10.0.0
matplotlib>=3.7.0
```
