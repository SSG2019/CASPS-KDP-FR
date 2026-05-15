import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussian_shaped_vq import GaussianShapedVectorQuantizer


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))


class ResBlock(nn.Module):
    """CIFAR 任务中使用的轻量残差块。"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualDecoderBlock(nn.Module):
    """重建解码器中的残差块。"""

    def __init__(self, in_channels, hidden_channels, residual_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, residual_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(residual_channels, hidden_channels, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, channels, num_layers, residual_channels):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualDecoderBlock(channels, channels, residual_channels) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)


class ClassificationEncoder(nn.Module):
    """分类任务编码器，输出 [B, 128, 4, 4] 潜变量。"""

    def __init__(self, latent_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ResBlock(256),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResBlock(512),
            nn.Conv2d(512, latent_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):
    """Bob 侧语义分类头。"""

    def __init__(self, latent_channels=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc(self.net(x))


class GaussianVQClassifier(nn.Module):
    """公开版分类模型：编码器 + 高斯整形 VQ + 分类头。"""

    def __init__(
        self,
        num_embeddings=16,
        embedding_dim=128,
        num_classes=10,
        commitment_cost=0.25,
        ot_weight=1.0,
    ):
        super().__init__()
        self.encoder = ClassificationEncoder(latent_channels=embedding_dim)
        self.quantizer = GaussianShapedVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ot_weight=ot_weight,
        )
        self.classifier = ClassificationHead(latent_channels=embedding_dim, num_classes=num_classes)

    def forward(self, x, channel=None):
        z = self.encoder(x)
        q = self.quantizer(z.float(), channel=channel)
        logits = self.classifier(q["quantized"])
        return {
            "logits": logits,
            "vq_loss": q["loss"],
            "perplexity": q["perplexity"],
            "encodings": q["encodings"],
            "target_hist": q["target_hist"],
        }


class ReconstructionEncoder(nn.Module):
    """重建任务编码器，输出 [B, 512, 4, 4] 特征。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ResBlock(256),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResBlock(512),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ReconstructionDecoder(nn.Module):
    """重建任务解码器，将 4x4 潜变量恢复为 32x32 图像。"""

    def __init__(self, num_residual_layers=4, residual_channels=32):
        super().__init__()
        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.residual_stack = ResidualStack(128, num_residual_layers, residual_channels)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.residual_stack(x)
        x = self.dec4(x)
        return self.out(x)


class GaussianVQReconstructor(nn.Module):
    """公开版重建模型：编码器 + 投影 + 高斯整形 VQ + 解码器。"""

    def __init__(
        self,
        num_embeddings=64,
        embedding_dim=256,
        commitment_cost=0.25,
        ot_weight=1.0,
        num_residual_layers=4,
        residual_channels=32,
    ):
        super().__init__()
        self.encoder = ReconstructionEncoder()
        self.pre_vq = nn.Conv2d(512, embedding_dim, kernel_size=1)
        self.quantizer = GaussianShapedVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            ot_weight=ot_weight,
        )
        self.post_vq = nn.Conv2d(embedding_dim, 512, kernel_size=1)
        self.decoder = ReconstructionDecoder(num_residual_layers, residual_channels)

    def forward(self, x, channel=None):
        z = self.pre_vq(self.encoder(x))
        q = self.quantizer(z.float(), channel=channel)
        recon = self.decoder(self.post_vq(q["quantized"]))
        return {
            "recon": recon,
            "vq_loss": q["loss"],
            "perplexity": q["perplexity"],
            "encodings": q["encodings"],
            "target_hist": q["target_hist"],
        }
