from cmath import log
import re
from urllib.parse import _ResultMixinStr
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

from models.nys_transformer import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        mu_token1, logvar_token1, mu_token2, logvar_token2, mu_token3, logvar_token3, mu_token4, logvar_token4, \
            mu_token5, logvar_token5, mu_token6, logvar_token6, feat_token = x[:, 0], x[:, 1], x[:, 2], x[:, 3], \
                x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9], x[:, 10], x[:, 11], x[:, 12:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((mu_token1.unsqueeze(1), logvar_token1.unsqueeze(1), mu_token2.unsqueeze(1), logvar_token2.unsqueeze(1), \
            mu_token3.unsqueeze(1), logvar_token3.unsqueeze(1), mu_token4.unsqueeze(1), logvar_token4.unsqueeze(1), mu_token5.unsqueeze(1), \
                logvar_token5.unsqueeze(1), mu_token6.unsqueeze(1), logvar_token6.unsqueeze(1), x), dim=1)
        return x

class PPEG_vib(nn.Module):
    def __init__(self, dim=512):
        super(PPEG_vib, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        mu_token, logvar_token, feat_token = x[:, 0], x[:, 1], x[:, 2:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((mu_token.unsqueeze(1), logvar_token.unsqueeze(1), x), dim=1)
        return x



class DVIBTrans(nn.Module):
    def __init__(self, feature_dim=512):
        super(DVIBTrans, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.muQuery1 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.sigmaQuery1 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.muQuery2 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.sigmaQuery2 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.muQuery3 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.sigmaQuery3 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.muQuery4 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.sigmaQuery4 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.muQuery5 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.sigmaQuery5 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.muQuery6 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.sigmaQuery6 = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.muQuery1, std=1e-6)
        nn.init.normal_(self.sigmaQuery1, std=1e-6)
        nn.init.normal_(self.muQuery2, std=1e-6)
        nn.init.normal_(self.sigmaQuery2, std=1e-6)
        nn.init.normal_(self.muQuery3, std=1e-6)
        nn.init.normal_(self.sigmaQuery3, std=1e-6)
        nn.init.normal_(self.muQuery4, std=1e-6)
        nn.init.normal_(self.sigmaQuery4, std=1e-6)
        nn.init.normal_(self.muQuery5, std=1e-6)
        nn.init.normal_(self.sigmaQuery5, std=1e-6)
        nn.init.normal_(self.muQuery6, std=1e-6)
        nn.init.normal_(self.sigmaQuery6, std=1e-6)
        
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self.fc = nn.Linear(feature_dim, 256)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        muQuery1 = self.muQuery1.expand(B, -1, -1).cuda()
        sigmaQuery1 = self.sigmaQuery1.expand(B, -1, -1).cuda()
        muQuery2 = self.muQuery2.expand(B, -1, -1).cuda()
        sigmaQuery2 = self.sigmaQuery2.expand(B, -1, -1).cuda()
        muQuery3 = self.muQuery3.expand(B, -1, -1).cuda()
        sigmaQuery3 = self.sigmaQuery3.expand(B, -1, -1).cuda()
        muQuery4 = self.muQuery4.expand(B, -1, -1).cuda()
        sigmaQuery4 = self.sigmaQuery4.expand(B, -1, -1).cuda()
        muQuery5 = self.muQuery5.expand(B, -1, -1).cuda()
        sigmaQuery5 = self.sigmaQuery5.expand(B, -1, -1).cuda()
        muQuery6 = self.muQuery6.expand(B, -1, -1).cuda()
        sigmaQuery6 = self.sigmaQuery6.expand(B, -1, -1).cuda()
        h = torch.cat((muQuery1, sigmaQuery1, muQuery2, sigmaQuery2, muQuery3, sigmaQuery3, \
            muQuery4, sigmaQuery4, muQuery5, sigmaQuery5, muQuery6, sigmaQuery6, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)

        h = self.fc(h)
        return h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4], h[:, 5], h[:, 6], \
            h[:, 7], h[:, 8], h[:, 9], h[:, 10], h[:, 11], h[:, 12:]

class VIBTrans(nn.Module):
    def __init__(self, feature_dim=512):
        super(VIBTrans, self).__init__()
        # Encoder
        self.pos_layer = PPEG_vib(dim=feature_dim)
        self.muQuery = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.muQuery, std=1e-6)
        nn.init.normal_(self.sigmaQuery, std=1e-6)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        # nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self.fc = nn.Linear(feature_dim, 256)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        muQuery = self.muQuery.expand(B, -1, -1).cuda()
        sigmaQuery = self.sigmaQuery.expand(B, -1, -1).cuda()
        h = torch.cat((muQuery, sigmaQuery, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)

        h = self.fc(h)
        return h[:, 0], h[:, 1], h[:, 2:]