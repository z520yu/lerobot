"""Visual Encoders for PLD Residual RL."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetV1Encoder(nn.Module):
    """
    Pretrained ResNet visual encoder.

    Uses ResNet18 as a lightweight encoder for visual features.
    """

    def __init__(
        self,
        output_dim: int = 256,
        freeze: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained ResNet18
        if pretrained:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet18(weights=None)

        # Remove final fc layer, keep up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.output_dim = output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: (B, C, H, W) or (B, N, C, H, W) for multi-camera

        Returns:
            features: (B, output_dim) or (B, N * output_dim)
        """
        if images.dim() == 5:
            # Multi-camera: (B, N, C, H, W) -> (B*N, C, H, W)
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            features = self.backbone(images).flatten(1)  # (B*N, 512)
            features = self.fc(features)  # (B*N, output_dim)
            features = features.view(B, N * self.output_dim)  # (B, N * output_dim)
        else:
            features = self.backbone(images).flatten(1)  # (B, 512)
            features = self.fc(features)  # (B, output_dim)
        return features


class SimpleConvEncoder(nn.Module):
    """
    Simple CNN encoder for smaller images.

    Lighter weight alternative to ResNet.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 256,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size (for 224x224 input)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 224, 224)
            conv_out_size = self.conv_layers(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.output_dim = output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: (B, C, H, W) or (B, N, C, H, W)

        Returns:
            features: (B, output_dim) or (B, N * output_dim)
        """
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            features = self.fc(self.conv_layers(images))
            features = features.view(B, N * self.output_dim)
        else:
            features = self.fc(self.conv_layers(images))
        return features
