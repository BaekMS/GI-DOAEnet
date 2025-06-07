import torch
from torch.nn.modules.activation import MultiheadAttention
from .util import channelwise_softmax_aggregation, LayerNorm

class Dualpath_block(torch.nn.Module):
    def __init__(self, num_heads, feature_size, rnn_layers=2):
        super(Dualpath_block, self).__init__()

        self.mhsa = MultiheadAttention(num_heads=num_heads, embed_dim=feature_size)
        self.rnn = torch.nn.GRU(input_size=2*feature_size, hidden_size=feature_size, bidirectional=False, batch_first=False, num_layers=rnn_layers)

        self.linear = torch.nn.Linear(feature_size, feature_size)

        self.norm_mhsa = LayerNorm(feature_size)
        self.norm_rnn = LayerNorm(feature_size)

        self.activation = torch.nn.ELU()

    def mhsa_forward(self, x, MPE):
        B, C, M, T = x.shape
        x_steered = x + MPE

        x_steered = x_steered.permute(0, 3, 1, 2).contiguous().view(B*T, C, M)
        x_steered = x_steered.permute(1, 0, 2) # C, B*T, M
        x_steered, _ = self.mhsa(x_steered, x_steered, x_steered)

        x_steered = x_steered.permute(1, 2, 0)  # B*T, C, M
        x_steered = x_steered.contiguous().view(B, T, C, M).permute(0, 2, 3, 1).contiguous().view(B*C, M, T)

        out = self.norm_mhsa(x_steered)
        out = out.view(B, C, M, T)
        
        return out

    def rnn_forward(self, x):
        B, C, M, T = x.shape
        out = x.clone()
        out=channelwise_softmax_aggregation(out, std=True)

        out = out.permute(2, 0, 1)  # T, B, 2M
        out=self.rnn(out)[0]  # T, B, M
        out = out.permute(1, 2, 0)  # B, M, T

        out=self.activation(out)
        out=self.linear(out.transpose(-2, -1)) # B, T, M
        out = self.norm_rnn(out.transpose(-2, -1))  # B, M, T   

        out = out.view(B, 1, M, T)  # B, 1, M, T
        return out
    
    def forward(self, x, MPE):

        out = self.mhsa_forward(x, MPE)  # Apply multi-head self-attention        
        x = x + out

        out = self.rnn_forward(x)  # Apply RNN
        x = x + out
        return x
    
class SpatioTemporal_block(torch.nn.Module):
    def __init__(self, n_heads, num_blocks, feature_size, rnn_layers):
        super(SpatioTemporal_block, self).__init__()     

        self.dualpath_block_list=torch.nn.ModuleList()  

        for i in range(num_blocks):    
            self.dualpath_block_list.append(
                Dualpath_block(
                    num_heads=n_heads,
                    feature_size=feature_size,
                    rnn_layers=rnn_layers))
            
    def forward(self, x, MPE):     
        B, C, M, T=x.shape
        MPE=MPE.unsqueeze(-1)
        MPE=torch.repeat_interleave(MPE, T, dim=-1)        
        for i in range(len(self.dualpath_block_list)):
            x = self.dualpath_block_list[i](x, MPE) 

        return x 