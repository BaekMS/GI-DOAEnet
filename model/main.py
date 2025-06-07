import torch
from .FFT import ConvSTFT
from .Channel_invariant_feature_extractor import Channel_invariant_feature_extractor
from .Microphone_positional_encoding import MicrophonePositionalEncoding
from .SpatioTemporal_block import SpatioTemporal_block
from .Spatial_spectrum_mapping import Spatial_spectrum_mapping
from .util import target_spatial_spectrum

class GI_DOAEnet(torch.nn.Module):
    def __init__(self, MPE_type):
        super(GI_DOAEnet, self).__init__()

        fft_len = 512
        feature_size = 128


        self.STFT=ConvSTFT(win_len = fft_len,
                          win_inc = 128,
                          fft_len = fft_len,
                          vad_threshold = 0.6666,
                          win_type = 'hann')
        
        self.CIFE=Channel_invariant_feature_extractor(init_feature = fft_len+2,
                                                      feature = feature_size, 
                                                      kernel_size = 3, 
                                                      padding = 1, 
                                                      stride = 1, 
                                                      dilation_rate = 2, 
                                                      num_blocks = 4,)

        self.MPE = MicrophonePositionalEncoding(feature=feature_size,
                                                MPE_type=MPE_type,
                                                alpha = 7,
                                                beta = 4)
        

        self.STDPBs=SpatioTemporal_block(n_heads=16,
                                        num_blocks=4,
                                        feature_size=feature_size,
                                        rnn_layers=2,)        
        
        self.SSMBs=Spatial_spectrum_mapping(feature = feature_size,
                                            degree_resolution = 1, 
                                            num_blocks = 3,
                                            kernel_size = 3, 
                                            dilation_rate = 2)        

        self.gammas = [2.5, 2.5, 2.5]

    def forward(self, x, mic_coordinate, vad=None, polar_position=None, return_target=False):
        
        # STFT
        x_stft_r, x_stft_i=self.STFT(x, cplx=True)
        x_stft=torch.cat([x_stft_r, x_stft_i], dim=2) # B, C, 2F, T
        
        # Channel invariant feature extraction
        x_feature=self.CIFE(x_stft) # B, C, M, T 

        # Microphone positional encoding
        MPE= self.MPE(mic_coordinate) # B, C, M
    
        # Spatio-temporal dual-path block
        x_spatio_temporal=self.STDPBs(x_feature, MPE) # B, C, M, T 

        # Spectrum mapping
        x_out=self.SSMBs(x_spatio_temporal)  # B, DS, Degree, T
        
        if return_target:
            vad_framed = self.STFT.get_vad_framed(vad)  
            target = target_spatial_spectrum(polar_position, vad_framed, self.SSMBs.degree_candidate, self.gammas)  # B, DS, Degree, T 

            return x_out, target  
        else:
            return x_out