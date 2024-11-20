import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F
from ..attention import SelfAttention, FeedForward
from .convnext import convnext_tiny




class PositionNet(nn.Module):
    def __init__(self, resize_input=448, in_dim=152, out_dim=768):
        super().__init__()
        
        self.resize_input = resize_input
        self.down_factor = 32 # determined by the convnext backbone 
        self.out_dim = out_dim
        assert self.resize_input % self.down_factor == 0
        
        # self.in_conv = nn.Conv2d(in_dim,3,3,1,1) # from num_sem to 3 channels
        self.in_conv = nn.Conv2d(4,3,3,1,1) # from num_sem to 3 channels
        self.convnext_tiny_backbone = convnext_tiny(pretrained=True)
        
        self.num_tokens = (self.resize_input // self.down_factor) ** 2
        
        convnext_feature_dim = 768
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_tokens, convnext_feature_dim).normal_(std=0.02))  # from BERT
      
        self.linears = nn.Sequential(
            nn.Linear( convnext_feature_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_feature = torch.nn.Parameter(torch.zeros([convnext_feature_dim]))


    def forward(self, img_A, label, mask):
        B = img_A.shape[0] 

        # print("img_A, label:", img_A.shape, label.shape)
        # token from edge map 
        img_A = torch.nn.functional.interpolate(img_A, self.resize_input, mode="nearest")
        label = torch.nn.functional.interpolate(label, self.resize_input, mode="nearest")
        
        sem = self.in_conv(torch.cat((img_A, label), dim=1))
        sem_feature = self.convnext_tiny_backbone(sem)
        objs = sem_feature.reshape(B, -1, self.num_tokens)
        objs = objs.permute(0, 2, 1) # N*Num_tokens*dim

        # expand null token
        null_objs = self.null_feature.view(1,1,-1)
        null_objs = null_objs.repeat(B,self.num_tokens,1)
        
        # mask replacing 
        mask = mask.view(-1,1,1)
        objs = objs*mask + null_objs*(1-mask)
        
        # add pos 
        objs = objs + self.pos_embedding

        # fuse them 
        objs = self.linears(objs)

        assert objs.shape == torch.Size([B,self.num_tokens,self.out_dim])        
        return objs, sem_feature



