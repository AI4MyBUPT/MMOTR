from cmath import log
import re
from urllib.parse import _ResultMixinStr
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

from models.model_transformer import DVIBTrans
from models.model_OT import OT_Construct
from utils.utils import NLLSurvLoss

class MCSP_OTMR(nn.Module):
    def __init__(self, use_condition=False, generator=False, g_model_type='OT', decoder_mode='specific', alpha_surv=0., wsi_encoding_dim=1024, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(MCSP_OTMR, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [wsi_encoding_dim, 256, 256], "big": [wsi_encoding_dim, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        self.wsi_encoding_dim = size[0]
        self.wsi_projection_dim = size[1]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks) 
        self.snn_layer = nn.Sequential(
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False),
        )
        self.omic_map_layer = nn.Sequential(
            nn.Linear(hidden[-1], hidden[-1])
        )
        ###### DVIB pathology compression
        self.use_condition = use_condition
        condition_dim, input_dim = size[0], size[1] 

        self.wsi_encoder = DVIBTrans(feature_dim=256)#self.wsi_encoding_dim)

        self.IBwsi_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        
        self.wsi_classifier1 = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.n_classes),
        )
        self.wsi_classifier2 = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.n_classes),
        )
        self.wsi_classifier3 = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.n_classes),
        )
        self.wsi_classifier4 = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.n_classes),
        )
        self.wsi_classifier5 = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.n_classes),
        )
        self.wsi_classifier6 = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.n_classes),
        )

        self.wsi_loss = NLLSurvLoss(alpha=alpha_surv)
        self.omic_loss = NLLSurvLoss(alpha=alpha_surv)
       

        #### OTCR missing genomic modality reconstruction
        self.generator = generator 
        if generator: 
            if g_model_type == 'OT':
                self.OT_Construct = OT_Construct(input_dim=input_dim, nfeats=len(omic_sizes))
            else: 
                raise NotImplementedError
        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)
        self.region_coattn = RegionMultiheadAttention(embed_dim=256, num_heads=1, region_num=8)

       

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=1)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=1)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

       
    
    def freeze_omic(self):
        for param in self.sig_networks.parameters():
            param.requires_grad = False

    def forward(self, omic_missing=False, **kwargs):
        all_loss = {}
        x_path = kwargs['x_path'] # (num_patch, 1024)
        # h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer
        x_path = self.wsi_net(x_path) ### path embeddings are fed through a FC layer

        if not omic_missing:
            x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)] 
            h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer   
            h_omic_bag_x = torch.stack(h_omic).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention)
        else:
            x_omic = None 
            h_omic_bag_x = None

        if kwargs['train']:       
            if self.generator:
                h_omic_bag_origin = h_omic_bag_x.detach() 

                mu_wsi1, logvar_wsi1, mu_wsi2, logvar_wsi2, mu_wsi3, logvar_wsi3, mu_wsi4, logvar_wsi4, \
                mu_wsi5, logvar_wsi5, mu_wsi6, logvar_wsi6, _ = self.wsi_encoder(x_path.unsqueeze(0))
                z_wsi1 = reparameterize(mu_wsi1, logvar_wsi1)
                z_wsi2 = reparameterize(mu_wsi2, logvar_wsi2)
                z_wsi3 = reparameterize(mu_wsi3, logvar_wsi3)
                z_wsi4 = reparameterize(mu_wsi4, logvar_wsi4)
                z_wsi5 = reparameterize(mu_wsi5, logvar_wsi5)
                z_wsi6 = reparameterize(mu_wsi6, logvar_wsi6)

                ##########fl
                z_wsi = torch.cat([z_wsi1, z_wsi2, z_wsi3, z_wsi4, z_wsi5, z_wsi6])
                results = self.OT_Construct(z_wsi.unsqueeze(1), h_omic_bag_origin)
                
                ##########
                
                h_omic_bag_y = results['recon_gene']
                h_path_ot = results['h_path_ot']
                h_omic_bag_y = h_omic_bag_y.permute(1,0,2)

                
                ############## init ############################
                recon_loss, align_loss, kl_wsi, kl_omic, kl_joint, kl_omic_componet = 0., 0., 0., 0., 0., 0.
                acc, risk_wsi, encode_wsi_loss = 0., 0., 0.

                ################## wsi dvib_x ######################
                A_wsi1, h_wsi1 = self.IBwsi_attention_head(z_wsi1)
                A_wsi2, h_wsi2 = self.IBwsi_attention_head(z_wsi2)
                A_wsi3, h_wsi3 = self.IBwsi_attention_head(z_wsi3)
                A_wsi4, h_wsi4 = self.IBwsi_attention_head(z_wsi4)
                A_wsi5, h_wsi5 = self.IBwsi_attention_head(z_wsi5)
                A_wsi6, h_wsi6 = self.IBwsi_attention_head(z_wsi6)
                A_wsi1 = torch.transpose(A_wsi1, 1, 0)
                A_wsi2 = torch.transpose(A_wsi2, 1, 0)
                A_wsi3 = torch.transpose(A_wsi3, 1, 0)
                A_wsi4 = torch.transpose(A_wsi4, 1, 0)
                A_wsi5 = torch.transpose(A_wsi5, 1, 0)
                A_wsi6 = torch.transpose(A_wsi6, 1, 0)
                h_wsi1 = torch.mm(F.softmax(A_wsi1, dim=1) , h_wsi1)
                h_wsi2 = torch.mm(F.softmax(A_wsi2, dim=1) , h_wsi2)
                h_wsi3 = torch.mm(F.softmax(A_wsi3, dim=1) , h_wsi3)
                h_wsi4 = torch.mm(F.softmax(A_wsi4, dim=1) , h_wsi4)
                h_wsi5 = torch.mm(F.softmax(A_wsi5, dim=1) , h_wsi5)
                h_wsi6 = torch.mm(F.softmax(A_wsi6, dim=1) , h_wsi6)

                logits_wsi1 = self.wsi_classifier1(h_wsi1)
                logits_wsi2 = self.wsi_classifier2(h_wsi2)
                logits_wsi3 = self.wsi_classifier3(h_wsi3)
                logits_wsi4 = self.wsi_classifier4(h_wsi4)
                logits_wsi5 = self.wsi_classifier5(h_wsi5)
                logits_wsi6 = self.wsi_classifier6(h_wsi6)

                label, c = kwargs['label'], kwargs['c']
                ### survival loss
                hazards_wsi1 = torch.sigmoid(logits_wsi1)
                hazards_wsi2 = torch.sigmoid(logits_wsi2)
                hazards_wsi3 = torch.sigmoid(logits_wsi3)
                hazards_wsi4 = torch.sigmoid(logits_wsi4)
                hazards_wsi5 = torch.sigmoid(logits_wsi5)
                hazards_wsi6 = torch.sigmoid(logits_wsi6)
                S_wsi1 = torch.cumprod(1 - hazards_wsi1, dim=1)
                S_wsi2 = torch.cumprod(1 - hazards_wsi2, dim=1)
                S_wsi3 = torch.cumprod(1 - hazards_wsi3, dim=1)
                S_wsi4 = torch.cumprod(1 - hazards_wsi4, dim=1)
                S_wsi5 = torch.cumprod(1 - hazards_wsi5, dim=1)
                S_wsi6 = torch.cumprod(1 - hazards_wsi6, dim=1)
                encode_wsi_loss1 = self.wsi_loss(hazards=hazards_wsi1, S=S_wsi1, Y=label, c=c)
                encode_wsi_loss2 = self.wsi_loss(hazards=hazards_wsi2, S=S_wsi2, Y=label, c=c)
                encode_wsi_loss3 = self.wsi_loss(hazards=hazards_wsi3, S=S_wsi3, Y=label, c=c)
                encode_wsi_loss4 = self.wsi_loss(hazards=hazards_wsi4, S=S_wsi4, Y=label, c=c)
                encode_wsi_loss5 = self.wsi_loss(hazards=hazards_wsi5, S=S_wsi5, Y=label, c=c)
                encode_wsi_loss6 = self.wsi_loss(hazards=hazards_wsi6, S=S_wsi6, Y=label, c=c)
                encode_wsi_loss = (encode_wsi_loss1+encode_wsi_loss2+encode_wsi_loss3+encode_wsi_loss4+encode_wsi_loss5+encode_wsi_loss6)/6
                risk_wsi = -torch.sum(S_wsi1, dim=1).detach().cpu().numpy() 

                max_kl_value = 2 #Prevent KL from being too large
                
                kl_wsi = (calc_kl_divergence(mu_wsi1, logvar_wsi1)+calc_kl_divergence(mu_wsi2, logvar_wsi2)+calc_kl_divergence(mu_wsi3, logvar_wsi3)+calc_kl_divergence(mu_wsi4, logvar_wsi4)+calc_kl_divergence(mu_wsi5, logvar_wsi5)+calc_kl_divergence(mu_wsi6, logvar_wsi6))/6- \
                    torch.clamp((calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi2, logvar_wsi2)+calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi3, logvar_wsi3)+ \
                        calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi4, logvar_wsi4)+calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi5, logvar_wsi5)+ \
                            calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi6, logvar_wsi6)+calc_kl_divergence(mu_wsi2, logvar_wsi2, mu_wsi3, logvar_wsi3)+ \
                                calc_kl_divergence(mu_wsi2, logvar_wsi2, mu_wsi4, logvar_wsi4)+calc_kl_divergence(mu_wsi2, logvar_wsi2, mu_wsi5, logvar_wsi5)+ \
                                    calc_kl_divergence(mu_wsi2, logvar_wsi2, mu_wsi6, logvar_wsi6)+calc_kl_divergence(mu_wsi3, logvar_wsi3, mu_wsi4, logvar_wsi4)+ \
                                        calc_kl_divergence(mu_wsi3, logvar_wsi3, mu_wsi5, logvar_wsi5)+calc_kl_divergence(mu_wsi3, logvar_wsi3, mu_wsi6, logvar_wsi6)+ \
                                            calc_kl_divergence(mu_wsi4, logvar_wsi4, mu_wsi5, logvar_wsi5)+calc_kl_divergence(mu_wsi4, logvar_wsi4, mu_wsi6, logvar_wsi6)+ \
                                                calc_kl_divergence(mu_wsi5, logvar_wsi5, mu_wsi6, logvar_wsi6))/15,max=max_kl_value)

                

                recon_loss = F.mse_loss(h_omic_bag_origin.permute(1,0,2), h_omic_bag_y, reduction='mean')
                flow_fit_loss = results['flow_fit_loss'] 
 
                all_loss['recon_loss'] =  recon_loss
                all_loss['encode_wsi_loss'] = encode_wsi_loss
                all_loss['kl_wsi'] = kl_wsi
                all_loss['risk_wsi'] = risk_wsi
                all_loss['flow_fit_loss'] = flow_fit_loss 

            else:
                h_omic_bag_origin = h_omic_bag_x.detach() 

                mu_wsi1, logvar_wsi1, mu_wsi2, logvar_wsi2, mu_wsi3, logvar_wsi3, mu_wsi4, logvar_wsi4, \
                mu_wsi5, logvar_wsi5, mu_wsi6, logvar_wsi6, _ = self.wsi_encoder(x_path.unsqueeze(0))
                z_wsi1 = reparameterize(mu_wsi1, logvar_wsi1)
                z_wsi2 = reparameterize(mu_wsi2, logvar_wsi2)
                z_wsi3 = reparameterize(mu_wsi3, logvar_wsi3)
                z_wsi4 = reparameterize(mu_wsi4, logvar_wsi4)
                z_wsi5 = reparameterize(mu_wsi5, logvar_wsi5)
                z_wsi6 = reparameterize(mu_wsi6, logvar_wsi6)

                ##########
                z_wsi = torch.cat([z_wsi1, z_wsi2, z_wsi3, z_wsi4, z_wsi5, z_wsi6])
                
                
                ############## init ############################
                recon_loss, align_loss, kl_wsi, kl_omic, kl_joint, kl_omic_componet = 0., 0., 0., 0., 0., 0.
                acc, risk_wsi, encode_wsi_loss = 0., 0., 0.

                ################## wsi dvib_x ######################
                A_wsi1, h_wsi1 = self.IBwsi_attention_head(z_wsi1)
                A_wsi2, h_wsi2 = self.IBwsi_attention_head(z_wsi2)
                A_wsi3, h_wsi3 = self.IBwsi_attention_head(z_wsi3)
                A_wsi4, h_wsi4 = self.IBwsi_attention_head(z_wsi4)
                A_wsi5, h_wsi5 = self.IBwsi_attention_head(z_wsi5)
                A_wsi6, h_wsi6 = self.IBwsi_attention_head(z_wsi6)
                A_wsi1 = torch.transpose(A_wsi1, 1, 0)
                A_wsi2 = torch.transpose(A_wsi2, 1, 0)
                A_wsi3 = torch.transpose(A_wsi3, 1, 0)
                A_wsi4 = torch.transpose(A_wsi4, 1, 0)
                A_wsi5 = torch.transpose(A_wsi5, 1, 0)
                A_wsi6 = torch.transpose(A_wsi6, 1, 0)
                h_wsi1 = torch.mm(F.softmax(A_wsi1, dim=1) , h_wsi1)
                h_wsi2 = torch.mm(F.softmax(A_wsi2, dim=1) , h_wsi2)
                h_wsi3 = torch.mm(F.softmax(A_wsi3, dim=1) , h_wsi3)
                h_wsi4 = torch.mm(F.softmax(A_wsi4, dim=1) , h_wsi4)
                h_wsi5 = torch.mm(F.softmax(A_wsi5, dim=1) , h_wsi5)
                h_wsi6 = torch.mm(F.softmax(A_wsi6, dim=1) , h_wsi6)

                logits_wsi1 = self.wsi_classifier1(h_wsi1)
                logits_wsi2 = self.wsi_classifier2(h_wsi2)
                logits_wsi3 = self.wsi_classifier3(h_wsi3)
                logits_wsi4 = self.wsi_classifier4(h_wsi4)
                logits_wsi5 = self.wsi_classifier5(h_wsi5)
                logits_wsi6 = self.wsi_classifier6(h_wsi6)

                label, c = kwargs['label'], kwargs['c']
                ### survival loss
                hazards_wsi1 = torch.sigmoid(logits_wsi1)
                hazards_wsi2 = torch.sigmoid(logits_wsi2)
                hazards_wsi3 = torch.sigmoid(logits_wsi3)
                hazards_wsi4 = torch.sigmoid(logits_wsi4)
                hazards_wsi5 = torch.sigmoid(logits_wsi5)
                hazards_wsi6 = torch.sigmoid(logits_wsi6)
                S_wsi1 = torch.cumprod(1 - hazards_wsi1, dim=1)
                S_wsi2 = torch.cumprod(1 - hazards_wsi2, dim=1)
                S_wsi3 = torch.cumprod(1 - hazards_wsi3, dim=1)
                S_wsi4 = torch.cumprod(1 - hazards_wsi4, dim=1)
                S_wsi5 = torch.cumprod(1 - hazards_wsi5, dim=1)
                S_wsi6 = torch.cumprod(1 - hazards_wsi6, dim=1)
                encode_wsi_loss1 = self.wsi_loss(hazards=hazards_wsi1, S=S_wsi1, Y=label, c=c)
                encode_wsi_loss2 = self.wsi_loss(hazards=hazards_wsi2, S=S_wsi2, Y=label, c=c)
                encode_wsi_loss3 = self.wsi_loss(hazards=hazards_wsi3, S=S_wsi3, Y=label, c=c)
                encode_wsi_loss4 = self.wsi_loss(hazards=hazards_wsi4, S=S_wsi4, Y=label, c=c)
                encode_wsi_loss5 = self.wsi_loss(hazards=hazards_wsi5, S=S_wsi5, Y=label, c=c)
                encode_wsi_loss6 = self.wsi_loss(hazards=hazards_wsi6, S=S_wsi6, Y=label, c=c)
                encode_wsi_loss = (encode_wsi_loss1+encode_wsi_loss2+encode_wsi_loss3+encode_wsi_loss4+encode_wsi_loss5+encode_wsi_loss6)/6
                risk_wsi = -torch.sum(S_wsi1, dim=1).detach().cpu().numpy() 

                max_kl_value = 2 #Prevent KL from being too large
                
                kl_wsi = (calc_kl_divergence(mu_wsi1, logvar_wsi1)+calc_kl_divergence(mu_wsi2, logvar_wsi2)+calc_kl_divergence(mu_wsi3, logvar_wsi3)+calc_kl_divergence(mu_wsi4, logvar_wsi4)+calc_kl_divergence(mu_wsi5, logvar_wsi5)+calc_kl_divergence(mu_wsi6, logvar_wsi6))/6- \
                    torch.clamp((calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi2, logvar_wsi2)+calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi3, logvar_wsi3)+ \
                        calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi4, logvar_wsi4)+calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi5, logvar_wsi5)+ \
                            calc_kl_divergence(mu_wsi1, logvar_wsi1, mu_wsi6, logvar_wsi6)+calc_kl_divergence(mu_wsi2, logvar_wsi2, mu_wsi3, logvar_wsi3)+ \
                                calc_kl_divergence(mu_wsi2, logvar_wsi2, mu_wsi4, logvar_wsi4)+calc_kl_divergence(mu_wsi2, logvar_wsi2, mu_wsi5, logvar_wsi5)+ \
                                    calc_kl_divergence(mu_wsi2, logvar_wsi2, mu_wsi6, logvar_wsi6)+calc_kl_divergence(mu_wsi3, logvar_wsi3, mu_wsi4, logvar_wsi4)+ \
                                        calc_kl_divergence(mu_wsi3, logvar_wsi3, mu_wsi5, logvar_wsi5)+calc_kl_divergence(mu_wsi3, logvar_wsi3, mu_wsi6, logvar_wsi6)+ \
                                            calc_kl_divergence(mu_wsi4, logvar_wsi4, mu_wsi5, logvar_wsi5)+calc_kl_divergence(mu_wsi4, logvar_wsi4, mu_wsi6, logvar_wsi6)+ \
                                                calc_kl_divergence(mu_wsi5, logvar_wsi5, mu_wsi6, logvar_wsi6))/15,max=max_kl_value)

                
                all_loss['encode_wsi_loss'] = encode_wsi_loss
                all_loss['kl_wsi'] = kl_wsi
                all_loss['risk_wsi'] = risk_wsi

            
            feat_wsi_norm = F.normalize(z_wsi, p=2, dim=1)
            feat_omic_norm = F.normalize(h_omic_bag_origin.squeeze(1), p=2, dim=1)
            cos_sim = F.cosine_similarity(feat_wsi_norm, feat_omic_norm, dim=1)
            loss_align = 1 - cos_sim.mean()
            all_loss['align'] = loss_align

            h_omic_bag = h_omic_bag_x
        else: 
            # inference 
            if self.generator:
                if h_omic_bag_x is not None:
                    mu_wsi1, logvar_wsi1, mu_wsi2, logvar_wsi2, mu_wsi3, logvar_wsi3, mu_wsi4, logvar_wsi4, \
                    mu_wsi5, logvar_wsi5, mu_wsi6, logvar_wsi6, _ = self.wsi_encoder(x_path.unsqueeze(0))
                    z_wsi1 = reparameterize(mu_wsi1, logvar_wsi1)
                    z_wsi2 = reparameterize(mu_wsi2, logvar_wsi2)
                    z_wsi3 = reparameterize(mu_wsi3, logvar_wsi3)
                    z_wsi4 = reparameterize(mu_wsi4, logvar_wsi4)
                    z_wsi5 = reparameterize(mu_wsi5, logvar_wsi5)
                    z_wsi6 = reparameterize(mu_wsi6, logvar_wsi6)

                    ##########fl
                    z_wsi = torch.cat([z_wsi1, z_wsi2, z_wsi3, z_wsi4, z_wsi5, z_wsi6])
                    results = self.OT_Construct(z_wsi.unsqueeze(1), h_omic_bag_x)
                    # h_omic_bag_origin = h_omic_bag_x.detach() 
                    

                    
                    h_omic_bag_y = results['recon_gene']
                    h_path_ot = results['h_path_ot'] 
                    h_omic_bag_y = h_omic_bag_y.permute(1,0,2)
                    h_omic_bag = h_omic_bag_x

                else: 
                    # missing genomic data
                    mu_wsi1, logvar_wsi1, mu_wsi2, logvar_wsi2, mu_wsi3, logvar_wsi3, mu_wsi4, logvar_wsi4, \
                    mu_wsi5, logvar_wsi5, mu_wsi6, logvar_wsi6, _ = self.wsi_encoder(x_path.unsqueeze(0))
                    z_wsi1 = reparameterize(mu_wsi1, logvar_wsi1)
                    z_wsi2 = reparameterize(mu_wsi2, logvar_wsi2)
                    z_wsi3 = reparameterize(mu_wsi3, logvar_wsi3)
                    z_wsi4 = reparameterize(mu_wsi4, logvar_wsi4)
                    z_wsi5 = reparameterize(mu_wsi5, logvar_wsi5)
                    z_wsi6 = reparameterize(mu_wsi6, logvar_wsi6)

                    ##########
                    z_wsi = torch.cat([z_wsi1, z_wsi2, z_wsi3, z_wsi4, z_wsi5, z_wsi6])
                    recon_gene, h_path_ot = self.OT_Construct.reconstruct(z_wsi.unsqueeze(1)) 
                    
                    h_omic_bag_y = recon_gene.permute(1,0,2)
                    h_omic_bag = h_omic_bag_y.permute(1,0,2) 
                    
            else:
                mu_wsi1, logvar_wsi1, mu_wsi2, logvar_wsi2, mu_wsi3, logvar_wsi3, mu_wsi4, logvar_wsi4, \
                mu_wsi5, logvar_wsi5, mu_wsi6, logvar_wsi6, _ = self.wsi_encoder(x_path.unsqueeze(0))
                z_wsi1 = reparameterize(mu_wsi1, logvar_wsi1)
                z_wsi2 = reparameterize(mu_wsi2, logvar_wsi2)
                z_wsi3 = reparameterize(mu_wsi3, logvar_wsi3)
                z_wsi4 = reparameterize(mu_wsi4, logvar_wsi4)
                z_wsi5 = reparameterize(mu_wsi5, logvar_wsi5)
                z_wsi6 = reparameterize(mu_wsi6, logvar_wsi6)

                ##########fl
                z_wsi = torch.cat([z_wsi1, z_wsi2, z_wsi3, z_wsi4, z_wsi5, z_wsi6])
                h_omic_bag = h_omic_bag_x
        
            # all_loss['noise_reg'] = noise_reg *1e-4
        ######

        
        
        
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, z_wsi.unsqueeze(1), z_wsi.unsqueeze(1))
        

        

        
        
        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()
        
        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()

        # if use_mask:
        #     if torch.rand(1) > 0.5:
        #         h_omic = torch.zeros_like(h_omic)        
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))
                
        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        # attention_scores = {'coattn': A_coattn, 'region_coattn': A_region_coattn, 'path': A_path, 'omic': A_omic}
        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        
        if self.generator:
            if kwargs['train']:
                return hazards, S, Y_hat, attention_scores, all_loss #, MI_wsi_loss, MI_omic_loss
            else:
                return hazards, S, Y_hat, attention_scores, all_loss
        path_scores = np.sum(np.abs(x_path.detach().cpu().numpy()), axis=1)
        gene_scores = np.sum(np.abs(h_omic_bag_x.detach().squeeze(1).cpu().numpy()), axis=1)
        return hazards, S, Y_hat, attention_scores, all_loss, (path_scores, gene_scores) #, MI_wsi_loss, MI_omic_loss


# Backward-compatible alias for older scripts/checkpoints.
Robust_MCAT = MCSP_OTMR

###
# ========== Modifying PyTorch Functionalities ======================
###
from torch.nn.functional import *

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    
    attn_output_weights_raw = attn_output_weights
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    
    if need_weights:
        if need_raw:
            
            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output,attn_output_weights_raw
            
            #attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            #return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_raw, attn_output_weights_raw.sum(dim=1) / num_heads
        else:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

def region_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    region_num: int,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    tgt_len, bsz, embed_dim = query.size() 
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    attn_output_weights = q @ k.transpose(1, 2)

    attn_output_weights_raw = attn_output_weights
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = attn_output_weights @ v
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    
    if need_weights:
        if need_raw:
            
            return attn_output,attn_output_weights_raw
        else:
            # average attention weights over heads
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module

class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, need_raw=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask)



class RegionMultiheadAttention(Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, region_num, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(RegionMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.region_num = region_num
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # MultiheadAttention
        import torch.nn as nn
        self.crossattn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=False)


    def padding(self,x):
        B, L, C = x.shape
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        _n = -H % self.region_num
        H, W = H+_n, W+_n

        region_size = int(H // self.region_num)
        region_num = self.region_num
        add_length = H * W - L
        x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        return x,H,W,add_length,region_num,region_size 

    def region_partition(self,x, region_size):
        """
        Args:
            x: (B, H, W, C)
            region_size (int): region size
        Returns:
            regions: (num_regions*B, region_size, region_size, C)
        """

        B, H, W, C = x.shape
        x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
        regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
        return regions

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, need_raw=True, attn_mask=None):
        #padding
        if torch.equal(key, value):
            key = key.permute(1,0,2)
            B, L, C = key.shape
            key,H,W,add_length,region_num,region_size = self.padding(key)
            key = key.view(B,H,W,C)
        # partition regions
        key_regions = self.region_partition(key, region_size)  # nW*B, region_size, region_size, C
        key_regions = key_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # value_regions = key_regions
        
        #R-attn
        
        outputs=[]
        for i in range(self.region_num**2):
            kv = key_regions[i,:]               # [n, 256]
            kv = kv.unsqueeze(1)     # [n, 1, 256]
            out, _ = self.crossattn(query, kv, kv)  # Q: [6,1,256], K/V: [n,1,256]
            outputs.append(out)
        attn_output = torch.cat(outputs, dim=0)      


        return attn_output, None
