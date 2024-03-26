import torch.nn as nn
import torch
from isegm.utils.serialization import serialize
from isegm.model.is_model import ISModel
from isegm.model.modifiers import LRMult
from isegm.model.mymodel.Encoder import OtherEncoder
from isegm.model.mymodel.Decoder import Decoder
from isegm.model.mymodel.Attention import SelfAttention, NewAttention
from isegm.model.ops import DistMaps

from isegm.model.modeling.deeplab_v3 import DeepLabV3Plus

class MutiModel(ISModel):
    @serialize
    def __init__(self, backbone='resnet50', deeplab_ch=256, aspp_dropout=0.5,
                 backbone_norm_layer=None, backbone_lr_mult=0.1, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)
        self.img_encoder = DeepLabV3Plus(backbone=backbone, ch=deeplab_ch, project_dropout=aspp_dropout,
                                         norm_layer=norm_layer, backbone_norm_layer=backbone_norm_layer)
        self.img_encoder.backbone.apply(LRMult(backbone_lr_mult))
        self.click_encoder = OtherEncoder(inpchannel=2, ch=deeplab_ch)

        self.imgattention = SelfAttention(inpchannel=deeplab_ch, oupchannel=deeplab_ch)
        self.clickattention = NewAttention(inpchannel=deeplab_ch, oupchannel=deeplab_ch, orichannel=2)
        self.prevattention = NewAttention(inpchannel=deeplab_ch, oupchannel=deeplab_ch, orichannel=1)

        self.prev_encoder = OtherEncoder(inpchannel=1, ch=deeplab_ch)
        self.decoder1 = Decoder(deeplab_ch)
        self.decoder2 = Decoder(deeplab_ch)
        self.decoder3 = Decoder(deeplab_ch)
        self.decoder3 = Decoder(deeplab_ch)
        self.prev_seg = nn.Sequential(
            nn.Conv2d(64, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True)
        )
        self.click_seg = nn.Sequential(
            nn.Conv2d(64, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True)
        )
        self.seg = nn.Sequential(
            nn.Conv2d(64, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True)
        )
        self.lastdistmap = DistMaps(norm_radius=20, spatial_scale=1.0,
                                cpu_mode=False, use_disks=True)

    def fistclick(self, image, points):

        image, prev_mask = self.prepare_input(image)
        click = self.dist_maps(image, points)

        limgfeat, imgfeat = self.img_encoder(image)
        lclickfeat, clickfeat = self.click_encoder(click, limgfeat)

        imgfeat = self.imgattention(imgfeat)
        clickfeat += self.clickattention(clickfeat, click)

        skipfeat2, feat2 = self.decoder2([lclickfeat[1], lclickfeat[0]], clickfeat)
        _, feat3 = self.decoder3(skipfeat2, imgfeat)

        click_put = self.click_seg(feat2)
        click_put = nn.functional.interpolate(click_put, scale_factor=2, mode='bilinear', align_corners=False)
        result = self.seg(feat3)
        result = nn.functional.interpolate(result, scale_factor=2, mode='bilinear', align_corners=False)

        return {'click_aux': click_put, 'instances': result, 'feat': clickfeat}
    #
    def otherclick(self, image, points):

        image, prev = self.prepare_input(image)
        click = self.dist_maps(image, points)

        limgfeat, imgfeat = self.img_encoder(image)
        lclickfeat, clickfeat = self.click_encoder(click, limgfeat)
        lprevfeat, prevfeat = self.prev_encoder(prev, lclickfeat)

        imgfeat = self.imgattention(imgfeat)
        clickfeat += self.clickattention(clickfeat, click)
        prevfeat += self.prevattention(prevfeat, prev)

        skipfeat1, feat1 = self.decoder1([lprevfeat[1], lprevfeat[0]], prevfeat)

        skipfeat2, feat2 = self.decoder2(skipfeat1, clickfeat)
        _, feat3 = self.decoder3(skipfeat2, imgfeat)

        prev_out = self.prev_seg(feat1)
        prev_out = nn.functional.interpolate(prev_out, scale_factor=2, mode='bilinear', align_corners=False)
        click_put = self.click_seg(feat2)
        click_put = nn.functional.interpolate(click_put, scale_factor=2, mode='bilinear', align_corners=False)
        result = self.seg(feat3)
        result = nn.functional.interpolate(result, scale_factor=2, mode='bilinear', align_corners=False)

        return {'prev_aux': prev_out,  'click_aux':click_put, 'instances': result, 'feat': prevfeat}
