import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.sync_batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Layer):
    def __init__(self, c1, c2):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2D(c1, c1 // 2, 1, bias_attr=False)
        self.conv2 = nn.Conv2D(c2, c2 // 2, 1, bias_attr=False)
        self.relu = nn.ReLU()
        
        self.last_conv = nn.Sequential(
            nn.Conv2D((c1 + c2) // 2, 256, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(256, 128, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(128, 64, kernel_size=1, stride=1)
        )

        self._init_weight()

    def forward(self, x, low_level_feat, factor):
        # 对 low_level_feat 进行卷积和 ReLU
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        # 对 x 进行卷积和 ReLU
        x = self.conv2(x)
        x = self.relu(x)

        # 双线性插值调整尺寸
        x = F.interpolate(x, size=[i * (factor // 2) for i in low_level_feat.shape[2:]], mode='bilinear', align_corners=True)
        if factor > 1:
            low_level_feat = F.interpolate(low_level_feat, size=[i * (factor // 2) for i in low_level_feat.shape[2:]], mode='bilinear', align_corners=True)

        # 拼接并通过最后的卷积层
        x = paddle.concat((x, low_level_feat), axis=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                # 使用 PaddlePaddle 的 Kaiming 初始化
                nn.initializer.KaimingNormal()(layer.weight)
            elif isinstance(layer, SynchronizedBatchNorm2d) or isinstance(layer, nn.BatchNorm2D):
                layer.weight.set_value(paddle.full(layer.weight.shape, 1.0))
                layer.bias.set_value(paddle.zeros_like(layer.bias))

# 原来的代码From SuperYOLO
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# class Decoder(nn.Module):
#     def __init__(self, c1,c2):
#         super(Decoder, self).__init__()
        
#         self.conv1 = nn.Conv2d(c1, c1//2, 1, bias=False)
#         self.conv2 = nn.Conv2d(c2, c2//2, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.last_conv = nn.Sequential(nn.Conv2d((c1+c2)//2, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.ReLU(),
#                                        nn.Conv2d(128, 64, kernel_size=1, stride=1))
#         self._init_weight()

#     def forward(self, x, low_level_feat,factor):
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.relu(low_level_feat) 

#         x = self.conv2(x)
#         x = self.relu(x) 
#         x = F.interpolate(x, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
#         if factor>1:
#             low_level_feat = F.interpolate(low_level_feat, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
#         x = torch.cat((x, low_level_feat), dim=1)
#         x = self.last_conv(x)

#         return x

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
