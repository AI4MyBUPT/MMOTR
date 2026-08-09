import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import reparameterize, poe, prior_expert
import ot

class OT_Attn_assem(nn.Module):
    def __init__(self,impl='pot-uot-l2',ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)
    
    def normalize_feature(self,x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (N, D)
            weight2 : (M, D)
        
        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()
            
            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
                                M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow 
            dist = torch.sum(dist)
            return flow, dist
        
        elif self.impl == "pot-uot-l2":
            a, b = torch.from_numpy(ot.unif(weight1.size()[0]).astype('float64')).to(weight1.device), torch.from_numpy(ot.unif(weight2.size()[0]).astype('float64')).to(weight2.device)
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach/cost_map_detach.max()
            
            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
                                M=M_cost.double(), reg=self.ot_reg,reg_m=self.ot_tau)
            flow = flow.type(torch.FloatTensor).cuda()
            
            dist = self.cost_map * flow # (N, M)
            dist = torch.sum(dist) # (1,) float
            return flow, dist
        
        else:
            raise NotImplementedError

    def forward(self,x,y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''
        x = x.squeeze()
        y = y.squeeze()
        
        x = self.normalize_feature(x)
        y = self.normalize_feature(y)
        
        pi, dist = self.OT(x, y)
        return pi.T.unsqueeze(0).unsqueeze(0), dist



class OT_Construct(nn.Module):
    def __init__(self, input_dim, nfeats=6):
        super(OT_Construct, self).__init__()
        self.results = {}
        ### OT-based Co-attention
        ot_reg=0.1
        ot_tau=0.5
        ot_impl="pot-uot-l2"
        self.OT_Match = OT_Attn_assem(impl=ot_impl,ot_reg=ot_reg,ot_tau=ot_tau)
        ### flow_fiter
        fc = [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Dropout(0.25),\
            nn.Linear(input_dim, nfeats), nn.ReLU(), nn.Dropout(0.25)]
        self.flow_fiter = nn.Sequential(*fc)
        ### decoder
        fc = [nn.Linear(input_dim, 2*input_dim), nn.ReLU(), nn.Dropout(0.25),\
            nn.Linear(2*input_dim, input_dim), nn.ReLU(), nn.Dropout(0.25)]
        self.decoder = nn.Sequential(*fc)

    def forward(self, h_path_bag, h_gene_bag):
        ### OT Recontructing 
        flow, _ = self.OT_Match(h_path_bag, h_gene_bag)
        flow_fit = self.flow_fiter(h_path_bag)
        flow_fit = flow_fit.permute((1,2,0)).unsqueeze(0)
        h_path_coattn = torch.mm(flow.squeeze(), h_path_bag.squeeze()).unsqueeze(1)
        self.results['h_path_ot'] = h_path_coattn
        recon_gene = self.decoder(h_path_coattn)


        self.results['recon_gene'] = recon_gene
        self.results['flow'] = flow 
        flow_fit_loss = F.mse_loss(flow, flow_fit, reduction='mean')
        self.results['flow_fit_loss'] = flow_fit_loss
        return self.results
    def reconstruct(self, h_path_bag):
        flow_fit = self.flow_fiter(h_path_bag)
        flow_fit = flow_fit.permute((1,2,0)).unsqueeze(0)
        h_path_coattn = torch.mm(flow_fit.squeeze(), h_path_bag.squeeze()).unsqueeze(1)
        h_path_ot = h_path_coattn
        recon_gene = self.decoder(h_path_coattn)

        return recon_gene, h_path_ot 