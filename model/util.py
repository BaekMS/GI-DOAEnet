import torch
from torch import nn
import math

def target_spatial_spectrum(target_polar_position, vad_framed, degrees, gammas):
    target_azimuths= target_polar_position[..., 0].unsqueeze(-1)  # B, Spk, 1
    cadidate_azimuths = torch.linspace(0, 360, degrees, device=target_azimuths.device).view(1, 1, -1)

    distance_abs = torch.abs(target_azimuths - cadidate_azimuths)  # B, Spk, num_candidate
    distance_abs = torch.stack((distance_abs, 360 - distance_abs), dim=0)  # 2, B, Spk, num_candidate
    distance = torch.min(distance_abs, dim=0).values  # B, Spk, num_candidate
    distance = torch.deg2rad(distance).unsqueeze(1)  # B, 1, Spk, num_candidate

    gammas = torch.tensor(gammas, dtype=torch.float32, device=target_azimuths.device).view(1, -1, 1, 1)  # 1, num_gamma, 1, 1
    gammas = torch.deg2rad(gammas)  # 1, num_gamma, 1, 1

    kappa = math.log(0.5**0.5) / (torch.cos(gammas) - 1)  # 1, num_gamma, 1, 1
    distance = torch.exp(kappa * (torch.cos(distance) - 1)).unsqueeze(-1)  # B, num_gamma, Spk, num_candidate, 1

    vad_framed = vad_framed.view(vad_framed.shape[0], 1, vad_framed.shape[1], 1, -1)  # B, 1, Spk, 1, T
    target = vad_framed * distance  # B, num_gamma, Spk, num_candidate, T
    target = torch.max(target, dim=2).values  # B, num_gamma, num_candidate, T
    return target  # B, num_gamma, num_candidate, T

def channelwise_softmax_aggregation(x, std=True):

    out_softmax=x.softmax(dim=1)
    out=x*out_softmax
  
    out_sum=out.sum(dim=1, keepdim=False)

    if std:
        out_std=out.std(dim=1, keepdim=False)
        out=torch.cat([out_sum, out_std], dim=1)
    else:
        out= out_sum
  
    return out

def cart2sph(x, y, z, is_degree=True):
    
    azimuth=torch.atan2(y, x)
    elevation=torch.pi/2-torch.atan2(z, torch.sqrt(x**2+y**2))
    distance=torch.sqrt(x**2+y**2+z**2)

    if is_degree:
        azimuth=torch.rad2deg(azimuth)
        elevation=torch.rad2deg(elevation)

    return azimuth, elevation, distance   

# this is from Asteroid: https://github.com/asteroid-team/asteroid

class LayerNorm(nn.Module):

    def __init__(self, feature_size):
        super(LayerNorm, self).__init__()
        self.feature_size = feature_size
        self.gamma = nn.Parameter(torch.ones(feature_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(feature_size), requires_grad=True)

    def forward(self, x, EPS: float = 1e-8):        

        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())
    
    def apply_gain_and_bias(self, normed_x):
 
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)
    

class ResidualBlock(nn.Module):

    def __init__(self,
                 feature,
                 kernel,
                 padding,
                 dilation=1,
                 norm='BN'):
        
        super(ResidualBlock, self).__init__()
        
        self.padding = (kernel - 1) * dilation
 
        self.pw_conv1d = nn.Conv1d(feature, feature, 1)
        self.dw_conv1d = nn.Conv1d(feature,
                                 feature,
                                 kernel,
                                 dilation=dilation,
                                 groups=feature,
                                 padding=self.padding)

        if norm=='BN':
            self.pw_norm=nn.BatchNorm1d(feature)
            self.dw_norm=nn.BatchNorm1d(feature)        
        elif norm=='LN':
            self.pw_norm=LayerNorm(feature)
            self.dw_norm=LayerNorm(feature)
        else:
            raise ValueError('Not exist normalization method')

        self.activation = nn.ELU()
 

    def forward(self, input):


        output=self.pw_conv1d(input)
        output=self.activation(output)
        output=self.pw_norm(output)

        output=self.dw_conv1d(output)  
        output=output[...,:-self.padding]  
        output=self.activation(output)
        output=self.dw_norm(output)

        output=output+input    
        
        return output

class ConvBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 norm='BN'):
        
        super(ConvBlock, self).__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1d=torch.nn.Conv1d(in_features,
                                    out_features,
                                    kernel_size,
                                    stride,
                                    padding=self.padding, 
                                    dilation=dilation,
                                    groups=groups)
        
        if norm=='BN':
            self.norm=torch.nn.BatchNorm1d(out_features)
        elif norm=='LN':
            self.norm=LayerNorm(out_features)
        else:
            raise ValueError('Not exist normalization method')
        
        self.activation=torch.nn.ELU()

    def forward(self, x):
        
        x=self.conv1d(x)
        x = x[..., :-self.padding]
        x=self.activation(x)
        x=self.norm(x)
        return x