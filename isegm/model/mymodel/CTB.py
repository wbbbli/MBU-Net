import torch.nn as nn
import torch
from isegm.utils.serialization import serialize
from isegm.model.is_model import ISModel
from isegm.model.modifiers import LRMult
from isegm.model.mymodel.Encoder import OtherEncoder
from isegm.model.mymodel.Decoder import Decoder
from isegm.model.mymodel.Attention import SelfAttention, NewAttention

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

        self.prev_encoder = OtherEncoder(inpchannel=1, ch=deeplab_ch)
        self.decoder1 = Decoder(deeplab_ch)
        self.decoder2 = Decoder(deeplab_ch)
        self.decoder3 = Decoder(deeplab_ch)
        self.seg = nn.Sequential(
            # nn.Conv2d(64 * 3, 64,
            #           kernel_size=3, padding=1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(64, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True)
        )

    def fistclick(self, image, points):

        image, prev_mask = self.prepare_input(image)
        click = self.dist_maps(image, points)

        limgfeat, imgfeat = self.img_encoder(image)
        lclickfeat, clickfeat = self.click_encoder(click, limgfeat)

        skipfeat2, feat2 = self.decoder2([lclickfeat[1], lclickfeat[0]], clickfeat)
        _, feat3 = self.decoder3(skipfeat2, imgfeat)
        feat = feat3
        result = self.seg(feat)
        result = nn.functional.interpolate(result, scale_factor=2, mode='bilinear', align_corners=False)

        return {'instances': result, 'feat': clickfeat}

    def otherclick(self, image, points):

        image, prev = self.prepare_input(image)
        click = self.dist_maps(image, points)

        limgfeat, imgfeat = self.img_encoder(image)
        lclickfeat, clickfeat = self.click_encoder(click, limgfeat)

        lprevfeat, prevfeat = self.prev_encoder(prev, lclickfeat)
        skipfeat1, feat1 = self.decoder1([lprevfeat[1], lprevfeat[0]], prevfeat)

        skipfeat2, feat2 = self.decoder2(skipfeat1, clickfeat)
        _, feat3 = self.decoder3(skipfeat2, imgfeat)

        feat = feat3
        result = self.seg(feat)
        result = nn.functional.interpolate(result, scale_factor=2, mode='bilinear', align_corners=False)

        return {'instances': result, 'feat': prevfeat}

    def backbone_forward(self, image, click, prev):

        limgfeat, imgfeat = self.img_encoder(image)
        lclickfeat, clickfeat = self.click_encoder(click, limgfeat)
        lprevfeat, prevfeat = self.prev_encoder(prev, lclickfeat)

        # imgfeat = self.img_attention(imgfeat)
        # clickfeat = self.click_attention(clickfeat, click)

        skipfeat1, feat1 = self.decoder1([lprevfeat[1], lprevfeat[0]], prevfeat)
        skipfeat2, feat2 = self.decoder2(skipfeat1, clickfeat)
        _, feat3 = self.decoder3(skipfeat2, imgfeat)

        # feat = torch.cat([feat1, feat2, feat3], dim=1)
        feat = feat3
        result = self.seg(feat)
        result = nn.functional.interpolate(result, scale_factor=2, mode='bilinear', align_corners=False)

        return {'instances': result}


if __name__ == '__main__':
    model = MutiModel(backbone='resnet34', deeplab_ch=128, aspp_dropout=0.20, with_prev_mask=True, use_leaky_relu=True, use_rgb_conv=False, use_disks=True, norm_radius=5)
    a = torch.ones((4, 3, 384, 384))
    b = torch.ones((4, 2, 384, 384))
    c = torch.ones((4, 3, 384, 384))
    d = model.backbone_forward(a, b)
