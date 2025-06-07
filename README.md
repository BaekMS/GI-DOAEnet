# Geometry-Invariant DOA Estimation Network
This repository contains the model code and pretrained weights for "_DNN-based Geometry-Invariant DOA Estimation with Microphone Positional Encoding and Complexity Gradual Training_" [[1]](#reference-1).

<img src="./figures/architecture.jpg" alt="Overall architecture" width="600"/>

Overall architecture of Geometry-Invariant DOA Estimation Network (GI-DOAEnet) with Microphone Positional Encoding (MPE). With $C$-channel signals and the coordinates of microphones, the geometry-invariant network structure estimates the azimuth.

## Microphone Positional Encoding (MPE)

<img src="./figures/v_eq.png" alt="v_eq">

- **Phase Modulation (PM)**:
<img src="./figures/PM_eq.png" alt="PM_eq">

- **Frequency Modulation (FM)**:
<img src="./figures/FM_eq.png" alt="FM_eq">

$M$ is the latent feature size. $r_{c}$, $\theta_{c}$, and $\phi_{c}$ are the distance, azimuth, and elevation angles of the $c$-th microphone, respectively. $\alpha$ is a amplitude scaling factor, and $\beta$ is a frequency scaling factor.

## Running the Code
You can infer the code by running by **python inference.py**. You can change the MPE type between **FM** and **PM** in the Python file.\
If you want to check the FLOPS and parameters of the model, uncomment the "count_flops_and_params" function in line 79 and change the device to "cpu".

## Examples
<table>
  <tr>
    <td align="center">
      <img src="./spectrum_plots/FM/10ch_0.png" alt="FM 10ch 0" width="200"/><br/>
      FM example with 10 channels and 1 speaker.
    </td>
    <td align="center">
      <img src="./spectrum_plots/PM/4ch_1.png" alt="PM 10ch 0" width="200"/><br/>
      PM example with 4 channels and 2 speakers.
    </td>
  </tr>
</table>

## References
<a name="reference-1"></a>
[1]  M.-S. Baek, J.-H. Chang and I. Cohen "DNN-based Geometry-Invariant DOA Estimation with Microphone Positional Encoding and Complexity Gradual Training," _IEEE Trans. Audio., Speech, Lang. Process._, 2025, doi: [10.1109/TASLPRO.2025.3577336](https://doi.org/10.1109/TASLPRO.2025.3577336).