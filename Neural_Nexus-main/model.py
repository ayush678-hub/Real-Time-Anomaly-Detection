import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class VideoClassifier(nn.Module):
    def __init__(self, num_classes=14, hidden_size=256, lstm_layers=1, dropout=0.3):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # backbone.features outputs (B, 1280, H', W') — correct feature extractor
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_out = 1280

        self.lstm = nn.LSTM(self.cnn_out, hidden_size, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.pool(self.cnn(x)).view(B, T, self.cnn_out)
        out, _ = self.lstm(feat)
        return self.classifier(out[:, -1, :])
