import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class ASPP(nn.Module):
    def __init__(self, out_channels=256):
        super(ASPP, self).__init__()
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            )
        self.layer6_2 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels , kernel_size=3, stride=1, padding=6,dilation=6, bias=True),
            nn.ReLU(),
            )
        self.layer6_3 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            )
        self.layer6_4 = nn.Sequential(
            nn.Conv2d(out_channels , out_channels , kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        feature_size = x.shape[-2:]
        global_feature = F.avg_pool2d(x, kernel_size=feature_size)

        global_feature = self.layer6_0(global_feature)

        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])
        out = torch.cat(
            [global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1)
        return out