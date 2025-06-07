import torch
from model.main import GI_DOAEnet
from glob import glob
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def plot_spectrum(output, target, file_name):
    png_dir = './spectrum_plots/' + file_name

    output = output.cpu().numpy()[0]
    target = target.cpu().numpy()[0]

    plt.figure(figsize=(6, 6))
    plt.rcParams['font.size'] = 8
    num_layers = output.shape[0]
    
    for i in range(num_layers):
        plt.subplot(num_layers, 2, 2*i + 1)
        plt.imshow(output[i], aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Output Layer {i + 1}')
        plt.xlabel('Frame')
        plt.ylabel('Azimuth')
        plt.colorbar()

        plt.subplot(num_layers, 2, 2*i + 2)
        plt.imshow(target[i], aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Target Layer {i + 1}')
        plt.xlabel('Frame')
        plt.ylabel('Azimuth')
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(png_dir, bbox_inches='tight', dpi=100, pad_inches=0.1)
    plt.close()
    plt.cla()
    plt.clf()

def to_tensor(data, device):
    data = data.to_numpy()
    data = np.stack(data, axis=0)  
    data = torch.from_numpy(data).to(device).float()  

    return data
    

def inference(model, device):
    # Load the data paths
    data_list = glob('./sample_input/**/*.pkl', recursive=True) 

    with torch.no_grad():
        model.eval()

        for data_path in tqdm(data_list, total=len(data_list), desc='Inference'):
            with open(data_path, 'rb') as f:
                data = pkl.load(f)

            # Unpack data
            input_audio = to_tensor(data['input_audio'], device)  
            vad = to_tensor(data['vad'], device)
            mic_coordinate = to_tensor(data['mic_coordinate'], device)
            polar_position = to_tensor(data['polar_position'], device) 

            # Forward pass through the model
            output, target = model(
                input_audio,
                mic_coordinate,
                vad,
                polar_position,                
                return_target=True
            )
            data_path = data_path.split('/')
            png_name = data_path[-2] + 'ch_' + data_path[-1].replace('.pkl', '.png')

            plot_spectrum(output, target, png_name)
            # Uncomment the line below to count FLOPs and parameters
            # count_flops_and_params(model, input_audio[..., :16000], vad[..., :16000], None, None) #

def count_flops_and_params(model, *inputs):
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    import contextlib

    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        flops = FlopCountAnalysis(model, inputs)
        flops = flops.total() / 1e9  # Convert to GFLOPs
        _, params=parameter_count_table(model)

    
    print(f'{"FLOPS:":<30} {flops:.2f}G')
    print(f'{"Parameters:":<30} {params}')

if __name__ == "__main__":
    ## Change the MPE_type to 'FM' or 'PM' as needed
    MPE_type = 'PM' # 'FM' or 'PM'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained = torch.load('./pretrained/GI_DOAEnet_{}.tar'.format(MPE_type), map_location='cpu')
    model = GI_DOAEnet(MPE_type=MPE_type)
    model.load_state_dict(pretrained, strict=True)  
    model.to(device)
    inference(model, device)

  