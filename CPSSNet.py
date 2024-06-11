import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .C_VSS import C_VSSBlock
from .P_VSS import P_VSSBlock
from .Patching import PatchEmbed2D,Final_PatchExpand2D


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class WS(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WS, self).__init__()
        self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
        self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

    def forward(self, x, res,ade):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2]*ade
        x = self.post_conv(x)
        return x

class CPSSNet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.1,
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=True,
                 patch_size=8,
                 num_classes=6,
                 use_aux_loss = True,
                 dim_scale = 8
                 ):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.patch_size = patch_size
        self.backbone = timm.create_model(model_name=backbone_name, features_only=True,pretrained=pretrained, output_stride=32, out_indices=(0, 1, 2,3))

        self.conv2 = ConvBN(192, decode_channels,kernel_size=1)
        self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(768, decode_channels, kernel_size=1)
        self.dim_scale = dim_scale

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.down = Conv(in_channels=3*decode_channels,out_channels=decode_channels,kernel_size=1)
        self.decoder_channels = decode_channels
        self.patchembed = PatchEmbed2D(patch_size=self.patch_size, in_chans=decode_channels, embed_dim=decode_channels)
        self.P_vssblock = P_VSSBlock(hidden_dim=decode_channels)
        self.final = Final_PatchExpand2D(dim=decode_channels,dim_scale=self.dim_scale)
        self.ws=WS(in_channels=decode_channels,decode_channels=decode_channels)
        self.wf=WF(in_channels=decode_channels,decode_channels=decode_channels)

        self.conv5 = nn.Conv2d(in_channels=3*decode_channels,out_channels=decode_channels,kernel_size=3,stride=2,padding=1)

        self.patchembed_L = PatchEmbed2D(patch_size=self.dim_scale, in_chans=2*decode_channels, embed_dim=2*decode_channels)
        self.C_vssblock = C_VSSBlock(hidden_dim=2*decode_channels)

        self.final_H = Final_PatchExpand2D(dim=decode_channels, dim_scale=self.dim_scale)

        self.final_W = Final_PatchExpand2D(dim=decode_channels, dim_scale=self.dim_scale)

        self.upconv_h = nn.Conv2d(in_channels=decode_channels//self.dim_scale,out_channels=decode_channels,kernel_size=1)
        self.upconv_w = nn.Conv2d(in_channels=decode_channels// self.dim_scale, out_channels=decode_channels,kernel_size=1)
        self.upconv_p = nn.Conv2d(in_channels=decode_channels// self.dim_scale, out_channels=decode_channels,kernel_size=1)

    def forward(self, x,imagename=None):

        h, w = x.size()[-2:]

        res1,res2,res3,res4 = self.backbone(x)

        res1h,res1w = res1.size()[-2:]

        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        middleres =torch.cat([res2,res3,res4],dim=1)
        middleres =self.conv5(middleres)
        middleres = F.interpolate(middleres,size=(self.decoder_channels,self.decoder_channels), mode='bicubic', align_corners=False)
        middlechannels_feature_H = middleres.permute(0,2,3,1)
        middlechannels_feature_W = middleres.permute(0,3,2,1)
        C_H_W = torch.cat([middlechannels_feature_H,middlechannels_feature_W],dim=1)

        f_l = self.patchembed_L(C_H_W)  ##(b,h,w,c)
        f_l = self.C_vssblock(f_l)
        f_h,f_w = f_l.chunk(2,dim=-1)
        f_h = self.final_H(f_h)
        f_h = f_h.permute(0,3,1,2)
        f_w = self.final_W(f_w)
        f_w = f_w.permute(0,3,2,1)
        f_h = self.upconv_h(f_h)
        f_w = self.upconv_w(f_w)
        f_c = f_h+f_w

        f = self.patchembed(middleres)
        f = self.P_vssblock(f)
        f = self.final(f)
        f = f.permute(0, 3, 1, 2)
        f = self.upconv_p(f)

        f = self.wf(f_c,f)

        res = F.interpolate(f,size=(res1h,res1w), mode='bicubic', align_corners=False)
        middleres= F.interpolate(middleres,size=(res1h,res1w),mode="bicubic",align_corners=False)

        res = self.ws(res,res1,middleres)

        res = self.segmentation_head(res)

        if self.training:
            if self.use_aux_loss == True:
                x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
                return x

            else:
                x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)

                return x
        else:
            x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
            return x

