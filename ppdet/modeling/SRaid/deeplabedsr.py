import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.sync_batchnorm import SynchronizedBatchNorm2d
from ppdet.modeling.SRaid.sr_decoder import Decoder
from ppdet.modeling.SRaid.edsr import EDSR
from ppdet.core.workspace import register, create

class EDSRConv(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2D(out_ch, out_ch, 3, padding=1),
        )

        self.residual_upsampler = nn.Sequential(
            nn.Conv2D(in_ch, out_ch, kernel_size=1, bias_attr=False),
        )

    def forward(self, input):
        return self.conv(input) + self.residual_upsampler(input)
    
@register
class DeepLab(nn.Layer):
    def __init__(self, ch, c1=128, c2=512, factor=2, sync_bn=True, freeze_bn=False):
        #ch 通道数  
        # c1/c2用于初始化Decoder?maybe it's Encoder in article     
        # factor放大倍数
        super(DeepLab, self).__init__()

        if sync_bn:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm2D

        self.sr_decoder = Decoder(c1, c2) 
        self.edsr = EDSR(num_channels=ch, input_channel=64, factor=8)  
        self.sr_upsample = DeconvSR(
            in_channels=64,     # Decoder 输出通道
            out_channels=ch     # SR 输出通道 (RGB+IR=4)
        )
        self.factor = factor

    def forward(self, low_level_feat, x):
        x_sr = self.sr_decoder(x, low_level_feat, self.factor)
        # x_sr_up = self.edsr(x_sr)
        x_sr_up = self.sr_upsample(x_sr)
        
        return x_sr_up
    
class DeconvSR(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv1 = nn.Sequential(
            nn.Conv2DTranspose(
                in_channels, in_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2D(in_channels),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2DTranspose(
                in_channels, in_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2D(in_channels),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2DTranspose(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            )
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


#原来的pytorch代码
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from models.sr_decoder_noBN_noD import Decoder
# from models.edsr import EDSR

# class EDSRConv(torch.nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(EDSRConv, self).__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             )

#         self.residual_upsampler = torch.nn.Sequential(
#             torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
#             )

#     def forward(self, input):
#         return self.conv(input)+self.residual_upsampler(input)


# class DeepLab(nn.Module):
#     def __init__(self, ch, c1=128, c2=512,factor=2, sync_bn=True, freeze_bn=False):
#         super(DeepLab, self).__init__()

#         if sync_bn == True:
#             BatchNorm = SynchronizedBatchNorm2d
#         else:
#             BatchNorm = nn.BatchNorm2d

#         self.sr_decoder = Decoder(c1,c2)
#         self.edsr = EDSR(num_channels=ch,input_channel=64, factor=8)
#         self.factor = factor


#     def forward(self, low_level_feat,x):
#         x_sr= self.sr_decoder(x, low_level_feat,self.factor)
#         x_sr_up = self.edsr(x_sr)

#         return x_sr_up