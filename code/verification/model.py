import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SiameseResNet18(nn.Module):
    def __init__(self, pretrained=False, normalize_embeddings=False):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        rgb = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(rgb.weight.mean(1, keepdim=True))
        self.backbone.fc = nn.Identity()
        self.normalize_embeddings = normalize_embeddings

    def encode(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (1,128,384):
            raise ValueError('Expected [B,1,128,384] grayscale images')
        z = self.backbone(image)
        return F.normalize(z, dim=1) if self.normalize_embeddings else z

    def forward(self, reference, query):
        # One shared backbone. Joint batch also avoids branch-order BatchNorm updates.
        z = self.encode(torch.cat((reference, query), dim=0))
        a,b = z.chunk(2, dim=0)
        return torch.linalg.vector_norm(a-b, dim=1)


def contrastive_loss(distance, label, margin=1.0):
    # Label 1 = genuine pair; label 0 = forgery query.
    return (label*distance.square() + (1-label)*F.relu(margin-distance).square()).mean()
