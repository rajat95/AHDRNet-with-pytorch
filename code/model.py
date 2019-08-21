import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch, 2*in_ch, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*in_ch, 4*in_ch, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*in_ch, 8*in_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out=self.conv(x)
        return out

class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
        )
    def forward(self, x1, x2):
        x=torch.cat([x1, x2], dim=1)
        x1=self.conv(x)
        out=torch.sigmoid(x1)
        return out

class DRDB(nn.Module):
    def __init__(self):
        super(DRDB, self).__init__()
        self.Dcov1=nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.Dcov2=nn.Conv2d(128, 96, 3, padding=2, dilation=2)
        self.Dcov3=nn.Conv2d(288, 128, 3, padding=2, dilation=2)
        self.conv=nn.Conv2d(608, 64, 1, padding=0)
    def forward(self, x):
        x1=self.Dcov1(x)
        x1=F.relu(x1)
        #print(x.shape, x1.shape)
        x1=torch.cat([x, x1], dim=1)
        x2=self.Dcov2(x1)
        x2=F.relu(x2)
        x2=torch.cat([x, x1, x2], dim=1)
        x3=self.Dcov3(x2)
        x3=F.relu(x3)
        x3=torch.cat([x, x1, x2, x3], dim=1)
        x4=self.conv(x3)
        out=F.relu(x4)
        return out

class AttentionNetwork(nn.Module):
    def __init__(self):
        super(AttentionNetwork, self).__init__()
        #self.encoder=Encoder(in_ch, out_ch)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(inplace=True),)
        self.attention=AttentionModule()

    def forward(self, x1, x2, x3):
        feature1=self.encoder(x1)
        refer=self.encoder(x2)
        feature2=self.encoder(x3)
        map1=self.attention(feature1, refer)
        map2=self.attention(feature2, refer)
        feature_1=torch.mul(feature1, map1)
        feature_2=torch.mul(feature2, map2)
        out=torch.cat([feature_1, refer, feature_2], dim=1)
        return out, refer

class MergingNetwork(nn.Module):
    def __init__(self):
        super(MergingNetwork, self).__init__()
        self.conv1=nn.Conv2d(192, 64, 3, padding=1)
        self.DRDB1=DRDB()
        self.DRDB2=DRDB()
        self.DRDB3=DRDB()
        self.conv2=nn.Conv2d(192, 64, 3, padding=1)
        self.conv3=nn.Conv2d(64, 32, 3, padding=1)
        self.conv4=nn.Conv2d(32, 3, 3, padding=1)
    def forward(self, x, xskip):
        x1=self.conv1(x)
        x1=F.relu(x1)
        x2=self.DRDB1(x1)
        x3=self.DRDB2(x2)
        x4=self.DRDB3(x3)
        x5=torch.cat([x2, x3, x4], dim=1)
        x6=self.conv2(x5)
        x6=F.relu(x6)
        x7=x6+xskip
        x8=self.conv3(x7)
        x8=F.relu(x8)
        x9=self.conv4(x8)
        out=F.relu(x9)
        return out

class AHDRNet(nn.Module):
    def __init__(self):
        super(AHDRNet, self).__init__()
        self.A=AttentionNetwork()
        self.M=MergingNetwork()
    def forward(self, x1, x2, x3):
        midout,ref=self.A(x1, x2, x3)
        finalout=self.M(midout, ref)
        return finalout








