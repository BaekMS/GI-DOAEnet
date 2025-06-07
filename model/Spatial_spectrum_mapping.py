import torch
from .util import channelwise_softmax_aggregation, ResidualBlock

class Spatial_spectrum_mapping(torch.nn.Module):
    def __init__(self, feature, degree_resolution, num_blocks, kernel_size=3, dilation_rate=2):
        super(Spatial_spectrum_mapping, self).__init__()

        self.degree_candidate=360//degree_resolution  

        self.ResidualConvBlocks=torch.nn.ModuleList()
        self.mapping_layers=torch.nn.ModuleList()

        for i in range(num_blocks):
            dilation=dilation_rate**i 

            self.ResidualConvBlocks.append(ResidualBlock(2*feature,
                                                         kernel_size,
                                                         dilation,
                                                         dilation,
                                                         norm='LN'))
            
            self.mapping_layers.append(torch.nn.Linear(2*feature, self.degree_candidate, bias=False))

    def forward(self, x, ):

        x=channelwise_softmax_aggregation(x, std=True)        
        
        outputs=[]

        for i in range(len(self.ResidualConvBlocks)):
            x=self.ResidualConvBlocks[i](x)
            out = self.mapping_layers[i](x.transpose(1, 2)).transpose(1, 2)
            outputs.append(out)

        outputs=torch.stack(outputs, dim=1).sigmoid()     
        
        return outputs # B, DS, Degree, T