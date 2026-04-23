# Neural-Nexus: AI for Social Good Surveillance

This project trains a video classification model for public-safety events using short CCTV clips. The goal is to classify each clip into one of 14 behavior categories and produce a severity score that helps prioritize response.

The current workflow is centered around a single notebook:

- `ai_for_social_goods_fixed.ipynb`

## Problem Scope

Given surveillance footage, the model should separate normal activity from risky events such as assault, theft, accidents, and vandalism. The system is designed for practical use under real-world constraints like variable video quality, noisy frames, and class imbalance.

## Classes

The dataset is organized into 14 folders (one class per folder):

1. abuse
2. arrest
3. arson
4. assault
5. burglary
6. explosion
7. fighting
8. normal
9. road accidents
10. robbery
11. shooting
12. shop lifting
13. stealing
14. vandalism

## Model Overview

The notebook uses a CNN + LSTM architecture:

1. MobileNetV2 extracts spatial features from each frame.
2. `TimeDistributed` applies the CNN across frame sequences.
3. LSTM layers learn temporal patterns across frames.
4. Dense layers produce final class probabilities.

Input setup in the notebook:

- Frames per video: 16
- Frame size: 112 x 112
- Channels: 3
- Batch size: 8

## Project Structure

Expected structure in your workspace:

```text
New folder/
    Dataset/
        abuse/
        arrest/
        ...
    Neural-Nexus/
        ai_for_social_goods_fixed.ipynb
        README.MD
```

Important: the notebook is configured to read `Dataset` from the parent workspace folder.

## Setup

1. Create and activate a Python virtual environment.
2. Open `ai_for_social_goods_fixed.ipynb` in VS Code.
3. Run the install cell (first code cell) to install dependencies.
4. Run cells in order from top to bottom.

Main libraries used:

- tensorflow
- opencv-python
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm

## Training Pipeline

The notebook follows this flow:

1. Path configuration and class definitions
2. Video frame extraction and preprocessing
3. Train/validation/test split
4. Data augmentation
5. Model build and training (two-phase fine-tuning)
6. Evaluation with confusion matrix and classification report
7. Severity scoring and alert-style output

## Outputs

When executed successfully, the notebook saves artifacts inside the repository directory:

- `processed_data/`
- `surveillance_model.keras`
- `logs/`
- Training/evaluation plots (for example, history and confusion matrix images)

Some preview assets are also available in the repository root from prior runs.


