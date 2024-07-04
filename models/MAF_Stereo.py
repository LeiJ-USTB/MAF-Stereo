from __future__ import print_function

import math
import timm
import torch.utils.data

from .submodule import *


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained = True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1, 2, 3, 5, 6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]


class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3] * 2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2] * 2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1] * 2, chans[1] * 2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, feat):
        x4, x8, x16, x32 = feat

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)

        return [x4, x8, x16, x32]


class MAF(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(MAF, self).__init__()
        self.semantic = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan // 2, cv_chan, 1))

        self.pre1 = nn.Sequential(
            BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cv_chan)
        )

        self.att_s = nn.Sequential(
            BasicConv(2 * cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cv_chan)
        )

        self.att_m = nn.Sequential(
            BasicConv(2 * cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            BasicConv(cv_chan, 2 * cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            BasicConv(2 * cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cv_chan)
        )

        self.att_l = nn.Sequential(
            BasicConv(2 * cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1,
                      padding=1),
            BasicConv(cv_chan, 2 * cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1,
                      padding=1),
            BasicConv(2 * cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1,
                      padding=1),
            BasicConv(cv_chan, 2 * cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1,
                      padding=1),
            BasicConv(2 * cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1,
                      padding=1),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cv_chan)
        )

        # self.att_m = nn.Sequential(
        #     BasicConv(2 * cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=5, stride=1, padding=2),
        #     nn.Conv3d(cv_chan, cv_chan, kernel_size=5, stride=1, padding=2, bias=False),
        #     nn.BatchNorm3d(cv_chan)
        # )

        # self.att_l = nn.Sequential(
        #     BasicConv(2 * cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=7, stride=1, padding=3),
        #     nn.Conv3d(cv_chan, cv_chan, kernel_size=7, stride=1, padding=3, bias=False),
        #     nn.BatchNorm3d(cv_chan)
        # )

        self.sigmoid = nn.Sigmoid()

        self.agg = nn.Sequential(
            BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cv_chan)
        )

        self.weight_init()

    def forward(self, x, residual):
        residual = self.semantic(residual).unsqueeze(2)
        y = x * residual
        y_r = self.pre1(y)

        xa = torch.cat((y, y_r), dim=1)
        xs = self.att_s(xa)
        xm = self.att_m(xa)
        xl = self.att_l(xa)
        x_all = xs + xm + xl
        wei = self.sigmoid(x_all)
        xi = y * wei + y_r * (1 - wei)

        xi = self.agg(xi)
        return xi


class hourglass_fusion(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels * 2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1), )

        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1))

        # self.MAF_8_d = MAF(in_channels * 2, 64)
        # self.MAF_16_d = MAF(in_channels * 4, 192)
        self.MAF_32 = MAF(in_channels * 6, 160)
        self.MAF_16 = MAF(in_channels * 4, 192)
        self.MAF_8 = MAF(in_channels * 2, 64)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)

        # conv1 = self.MAF_8_d(conv1, imgs[1])
        conv2 = self.conv2(conv1)

        # conv2 = self.MAF_16_d(conv2, imgs[2])
        conv3 = self.conv3(conv2)

        conv3 = self.MAF_32(conv3, imgs[3])
        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2 = self.MAF_16(conv2, imgs[2])
        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv1 = self.MAF_8(conv1, imgs[1])
        conv = self.conv1_up(conv1)

        return conv

class FeatureAtt(SubModule):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1))
        
        self.weight_init()

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att) * cv
        return cv

class MAF_Stereo(nn.Module):
    def __init__(self, maxdisp):
        super(MAF_Stereo, self).__init__()
        self.maxdisp = maxdisp
        self.feature = Feature()
        self.feature_up = FeatUp()

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.agg = FeatureAtt(8, 96)
        self.hourglass_fusion = hourglass_fusion(8)
        self.corr_stem = BasicConv(4, 8, is_3d=True, kernel_size=3, stride=1, padding=1)

    def forward(self, left, right):
        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left = self.feature_up(features_left)
        features_right = self.feature_up(features_right)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        corr_volume = build_gwc_volume_norm_best(match_left, match_right, self.maxdisp // 4, 4)
        corr_volume = self.corr_stem(corr_volume)
        volume = self.agg(corr_volume, features_left[0])
        cost = self.hourglass_fusion(volume, features_left).squeeze(1)

        # corr_volume = build_gwc_volume(match_left, match_right, self.maxdisp // 4, 8)
        # cost = self.hourglass_fusion(corr_volume, features_left).squeeze(1)

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        _, ind = cost.sort(1, True)
        pool_ind = ind[:, :2]
        cost = torch.gather(cost, 1, pool_ind)
        prob = F.softmax(cost, 1)
        pred = torch.sum(prob * pool_ind, dim=1, keepdim=True)

        pred_up = context_upsample(pred, spx_pred.float())

        if self.training:
            return [pred_up * 4, pred.squeeze(1) * 4]

        else:
            return [pred_up * 4]
