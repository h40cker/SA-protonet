import torch.nn as nn
import torch
import torch.nn.functional as F
class EnhancedProtoNet(nn.Module):
    """集成优化的ProtoNet"""

    def __init__(self, in_channels=3, hid_dim=64, out_dim=640):
        super(EnhancedProtoNet, self).__init__()
        self.encoder = ResNet12(in_channels, hid_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        return self.encoder(x)

    @staticmethod
    def compute_prototypes(support_features, support_labels):
        unique_labels = torch.unique(support_labels)
        prototypes = torch.stack([
            support_features[support_labels == label].mean(0)
            for label in unique_labels
        ])
        return prototypes

    def classify(self, query_features, prototypes):
        dists = torch.cdist(query_features, prototypes)
        return -dists  # 负距离作为logits
class ResNet12(nn.Module):
    """优化后的骨干网络"""

    def __init__(self, in_channels=3, hid_dim=64, out_dim=640):
        super(ResNet12, self).__init__()
        self.in_channels = hid_dim
        self.conv1 = nn.Conv2d(in_channels, hid_dim, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hid_dim)
        self.layer1 = self._make_layer(hid_dim, stride=1)
        self.layer2 = self._make_layer(hid_dim * 2, stride=2)
        self.layer3 = self._make_layer(hid_dim * 4, stride=2)
        self.layer4 = self._make_layer(hid_dim * 8, stride=2)
        self.dropblock = DropBlock()  # 加入DropBlock
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def _make_layer(self, out_channels, stride):
        layers = [SEResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.append(SEResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.dropblock(x)  # 应用DropBlock
        x = self.layer2(x)
        x = self.dropblock(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)
class SEResidualBlock(nn.Module):
    """带SE模块的残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # 加入SE注意力
        out += self.shortcut(x)
        return F.relu(out)
class SEBlock(nn.Module):
    """通道注意力机制"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DropBlock(nn.Module):
    """结构化Dropout，增强泛化能力"""

    def __init__(self, block_size=3, drop_prob=0.1):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        # 简化版实现：实际需按论文计算掩码
        mask = torch.ones_like(x) if torch.rand(1) > self.drop_prob else torch.zeros_like(x)
        return x * mask
