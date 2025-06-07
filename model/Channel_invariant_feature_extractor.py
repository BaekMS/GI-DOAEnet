import torch
from .util import ConvBlock, ResidualBlock

class Channel_invariant_feature_extractor(torch.nn.Module):
    def __init__(self, init_feature, feature, kernel_size = 3, padding = 1, stride = 1, dilation_rate = 2, num_blocks = 4):   
        super(Channel_invariant_feature_extractor, self).__init__()       


        self.init_BN=torch.nn.BatchNorm1d(init_feature)     
        self.init_ConvBlock=ConvBlock(init_feature, feature, kernel_size, stride, padding, norm = 'BN')


        self.ResidualConvBlocks=torch.nn.ModuleList()

        for i in range(num_blocks):
            dilation=dilation_rate**i 
            self.ResidualConvBlocks.append(ResidualBlock(feature, 
                                                        kernel_size,
                                                        dilation, 
                                                        dilation,
                                                        norm='BN'))    
    def forward(self, x):
        B, C, F, T=x.shape

        x=x.view(B*C, F, T)
        x=self.init_BN(x)
        x=self.init_ConvBlock(x)

        for i in range(len(self.ResidualConvBlocks)):
            x=self.ResidualConvBlocks[i](x)
        
        x=x.view(B, C, x.shape[1], T)
        
        return x  # B, C, M, T

        