# this is from torch-stft: https://github.com/pseeth/torch-stft
# this is from ConferencingSpeech2021: https://github.com/ConferencingSpeech/ConferencingSpeech2021

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch as th
import torch
import numpy as np
from scipy.signal import get_window

EPSILON = th.finfo(th.float32).eps
MATH_PI = math.pi

def init_kernels(win_len,
                 win_inc,
                 fft_len,
                 win_type=None,
                 invers=False):
    if win_type == 'None' or win_type is None:
        # N 
        window = np.ones(win_len)
    else:
        # N
        window = get_window(win_type, win_len, fftbins=True)#**0.5
   
    N = fft_len
    # N x F
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    # N x F
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    # 2F x N
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    # 2F x N * N => 2F x N
    kernel = kernel*window
    # 2F x 1 x N
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, 
                 win_len,
                 win_inc,
                 fft_len=None,
                 vad_threshold=2/3,
                 win_type='hamming',
                #  fix=True
                 ):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        # 2F x 1 x N
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        vad_kernel=torch.ones((1,1, self.fft_len), dtype=torch.float32)/self.fft_len
        self.register_buffer('vad_kernel', vad_kernel)
                
        self.register_buffer('weight', kernel)
        self.vad_threshold=vad_threshold
        
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    
    def get_vad_framed(self, vad):
        N, P, L=vad.shape
        vad=vad.view(N*P, 1, L)
        pad_size=(self.stride-L)%self.stride
        vad=F.pad(vad, [0, pad_size])
        vad=F.conv1d(vad, self.vad_kernel, stride=self.stride)
        vad=vad.view(N, P, -1).ge(self.vad_threshold).long()
        return vad

    def forward(self, inputs,  cplx=True):
        
        
        if inputs.dim() == 2:
            # N x 1 x L
            inputs = torch.unsqueeze(inputs, 1)
            pad_size=(self.stride-L)%self.stride
            inputs=F.pad(inputs, [0, pad_size])

            # N x 2F x T
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)
            # N x F x T
            r, i = th.chunk(outputs, 2, dim=1)
        else:
            
            N, C, L = inputs.shape
            inputs = inputs.view(N * C, 1, L)

            pad_size=(self.stride-L)%self.stride
            inputs=F.pad(inputs, [0, pad_size])
          
            
            # NC x 2F x T
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)

            # N x C x 2F x T
            outputs = outputs.view(N, C, -1, outputs.shape[-1])
            
            # N x C x F x T
            r, i = th.chunk(outputs, 2, dim=2)
        
     
        if cplx:
            return r, i
        else:
            mags = th.clamp(r**2 + i**2, EPSILON)**0.5
            phase = th.atan2(i, r)
            return mags, phase
