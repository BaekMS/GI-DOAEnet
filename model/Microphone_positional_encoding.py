import torch
import numpy as np
from .util import cart2sph

class MicrophonePositionalEncoding(torch.nn.Module):
    def __init__(self, feature, MPE_type, alpha=7, beta=4): 
        super(MicrophonePositionalEncoding, self).__init__()

        assert MPE_type in ['FM', 'PM'], "MPE_type must be either 'FM' or 'PM'."       

        self.v = np.linspace(0, 1, feature//4, dtype=np.float32, endpoint=False)
        self.v = torch.from_numpy(self.v).view(1, 1, -1)
        self.v = torch.nn.Parameter(self.v, requires_grad=False)

        self.MPE_type = MPE_type
        self.alpha = alpha
        self.beta = beta

    def forward(self, mic_coordinate): 

        azimuth, elevation, distance = cart2sph(mic_coordinate[..., 0], mic_coordinate[..., 1], mic_coordinate[..., 2], is_degree=False)

        azimuth = azimuth.unsqueeze(-1)  # B, C, 1
        elevation = elevation.unsqueeze(-1)
        distance = distance.unsqueeze(-1)

        pe_list = []

        if self.MPE_type == 'FM':
            pe_list.append(torch.cos(self.beta * azimuth * self.v))
            pe_list.append(torch.sin(self.beta * azimuth * self.v))
            pe_list.append(torch.cos(self.beta * elevation * self.v))
            pe_list.append(torch.sin(self.beta * elevation * self.v))
        elif self.MPE_type == 'PM':
            pe_list.append(torch.cos(azimuth + 2 * torch.pi * self.beta * self.v))
            pe_list.append(torch.sin(azimuth + 2 * torch.pi * self.beta * self.v))
            pe_list.append(torch.cos(elevation + 2 * torch.pi * self.beta * self.v))
            pe_list.append(torch.sin(elevation + 2 * torch.pi * self.beta * self.v))
        
        pe = distance * self.alpha * torch.cat(pe_list, dim=-1)

        return pe