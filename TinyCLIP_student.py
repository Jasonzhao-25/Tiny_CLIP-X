import torch.nn as nn
import math


class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs(math.log(channel, 2) + b) / gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class TinyCLIPStudent(nn.Module):
    def __init__(self, embed_dim=512, num_classes=20, freeze_layers=0, use_eca_net=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        self.backbone_features = nn.Sequential(
            self.conv1, self.relu1,
            self.conv2, self.relu2
        )

        self.use_eca_net = use_eca_net
        if self.use_eca_net:
            self.eca_block = ECABlock(channel=128)
        else:
            self.eca_block = None

        if freeze_layers > 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        if freeze_layers > 1:
            for param in self.conv2.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(128, 512)
        self.classifier = nn.Linear(128, num_classes)
        self.auxiliary_head = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False, return_logits=False, return_aux_logits=False):
        feature_map_raw = self.backbone_features(x)
        if self.eca_block is not None:
            feature_map = self.eca_block(feature_map_raw)
        else:
            feature_map = feature_map_raw

        pooled = self.pool(feature_map)
        flat = pooled.view(pooled.size(0), -1)
        embedding = self.projector(flat)

        outputs = [embedding]
        if return_features:
            outputs.append(feature_map)
        if return_logits:
            logits = self.classifier(flat)
            outputs.append(logits)
        if return_aux_logits:
            aux_logits = self.auxiliary_head(flat)
            outputs.append(aux_logits)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]