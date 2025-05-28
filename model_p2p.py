#model_p2p.py
"""
Model definitions including VGG19 Encoder, FPN Decoder, ASPP, and Coordinate Head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import from config
from config import PSF_HEAD_TEMP, MODEL_INPUT_SIZE # MODEL_INPUT_SIZE not directly used here for layer defs

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module."""
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]): # Default rates from DeepLab
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        for rate in rates[1:]: # Start from rates[1] because rate=0 is 1x1 conv
             self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                        padding=rate, dilation=rate, bias=False))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.bn_ops = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(len(self.convs) + 1)])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(self.convs) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2) 
        )

    def forward(self, x):
        size = x.shape[2:]
        features = []
        for i, conv in enumerate(self.convs):
            features.append(F.relu(self.bn_ops[i](conv(x))))
        gap_feat = self.global_pool(x)
        gap_feat = F.interpolate(gap_feat, size=size, mode='bilinear', align_corners=False)
        features.append(self.bn_ops[-1](gap_feat))
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x

class VGG19Encoder(nn.Module):
    def __init__(self):
        super(VGG19Encoder, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = list(vgg19.features)
        self.feature_layers = nn.ModuleList(features)
        self.capture_indices = {3, 8, 17, 26, 35} # C1, C2, C3, C4, C5 before last pool

    def forward(self, x):
        results = {}
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.capture_indices:
                 if i == 3: results['C1'] = x
                 elif i == 8: results['C2'] = x
                 elif i == 17: results['C3'] = x
                 elif i == 26: results['C4'] = x
                 elif i == 35: results['C5'] = x
        return [results['C1'], results['C2'], results['C3'], results['C4'], results['C5']]

class SmallPSFEncoder(nn.Module):
    def __init__(self):
        super(SmallPSFEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=1), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.encoder(x)

class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 128, 256, 512, 512], fpn_channels=256, out_channels=64):
        super(FPNDecoder, self).__init__()
        assert len(encoder_channels) == 5
        self.lateral_convs = nn.ModuleList()
        for enc_ch in reversed(encoder_channels):
            self.lateral_convs.append(nn.Conv2d(enc_ch, fpn_channels, kernel_size=1))
        self.smooth_convs = nn.ModuleList()
        for _ in range(len(encoder_channels)):
             self.smooth_convs.append(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1))
        self.final_conv = nn.Conv2d(fpn_channels, out_channels, kernel_size=3, padding=1)

    def _upsample_add(self, top_down_feat, lateral_feat):
        _, _, H, W = lateral_feat.shape
        upsampled_feat = F.interpolate(top_down_feat, size=(H, W), mode='bilinear', align_corners=False)
        return upsampled_feat + lateral_feat

    def forward(self, x_top, encoder_features_c1_c4):
        C1, C2, C3, C4 = encoder_features_c1_c4
        all_features = [C1, C2, C3, C4, x_top]
        pyramid_features = []
        p = self.lateral_convs[0](all_features[-1])
        p = self.smooth_convs[0](p)
        pyramid_features.append(p)
        for i in range(1, len(self.lateral_convs)):
            lateral_idx = len(all_features) - 1 - i
            lateral_feat = self.lateral_convs[i](all_features[lateral_idx])
            p_prev = pyramid_features[-1]
            top_down_feat = self._upsample_add(p_prev, lateral_feat)
            p = self.smooth_convs[i](top_down_feat)
            pyramid_features.append(p)
        p1_output = pyramid_features[-1]
        out = F.relu(self.final_conv(p1_output))
        return out

class CoordinateHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, out_dim=2):
        super(CoordinateHead, self).__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear((in_channels // 4), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid() # Add Sigmoid to output values between 0 and 1
        )

    def forward(self, x):
        x = self.conv_reduce(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        coords = self.fc(x)
        return coords

class VGG19FPNASPP(nn.Module):
    def __init__(self):
        super(VGG19FPNASPP, self).__init__()
        self.image_encoder = VGG19Encoder()
        self.mask_encoder = SmallPSFEncoder()

        vgg_c1_ch, vgg_c2_ch, vgg_c3_ch, vgg_c4_ch, vgg_c5_ch = 64, 128, 256, 512, 512
        mask_feat_ch = 64
        fusion_in_ch_c5 = vgg_c5_ch + mask_feat_ch
        fusion_out_ch_c5 = 512

        self.fusion_conv_c5 = nn.Sequential(
            nn.Conv2d(fusion_in_ch_c5, fusion_out_ch_c5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_out_ch_c5), nn.ReLU(inplace=True)
        )
        self.aspp_c5 = ASPP(in_channels=fusion_out_ch_c5, out_channels=fusion_out_ch_c5)
        
        fpn_encoder_channels = [vgg_c1_ch, vgg_c2_ch, vgg_c3_ch, vgg_c4_ch, fusion_out_ch_c5]
        self.fpn_decoder = FPNDecoder(
             encoder_channels=fpn_encoder_channels,
             fpn_channels=256, out_channels=64 # FPN output channels = 64
        )
        # Use the new CoordinateHead
        self.coordinate_head = CoordinateHead(in_channels=64, hidden_dim=128, out_dim=2) # FPN out_channels is 64

    def forward(self, image, mask):
        if mask.dim() == 3: mask = mask.unsqueeze(1)

        encoder_features = self.image_encoder(image)
        C1, C2, C3, C4, C5 = encoder_features
        mask_features = self.mask_encoder(mask)

        fused_features = torch.cat([C5, mask_features], dim=1)
        fused_c5 = self.fusion_conv_c5(fused_features)
        aspp_output = self.aspp_c5(fused_c5)
        
        decoder_output = self.fpn_decoder(aspp_output, [C1, C2, C3, C4]) # (B, 64, H, W)
        
        # Predict coordinates
        coords = self.coordinate_head(decoder_output) # (B, 2)
        return coords
