\# CSI Channel Dataset — 5G NR (8Tx × 4Rx, CDL-C)



Dataset of MIMO channel matrices generated via 5G NR simulation with CDL-C model,

developed for research in CSI feedback compression with autoencoders.

\---



\## Download do Dataset



The dataset (\~14 GB) is available on Google Drive.:



\*\*\[📥 Download — Google Drive](https://drive.google.com/file/d/19wJ4j2sLZ5D9avH7cVqrgYOzmltbx4lJ/view?usp=drive\_link)\*\*



> The dataset is \*\*not included in this repository\*\* due to the size of the '.mat' files.



\---



\## System Parameters



| Parameter           | Value         |

|---------------------|---------------|

| Channel Model       | CDL-C         |

| Tx Antennas         | 8 (gNB)       |

| Rx Antennas         | 4 (UE)        |

| Number of samples   | 20.000        |

| Output format       | `.mat`        |



\---



\## Dataset Content



Each `.mat` file contains:



\- \*\*`H`\*\* — Complex channel matrices, pre-processed from  `\[624 14 Nrx × Ntx]` to `\[28 × Ntx x 2]`

\- \*\*`H\_norm`\*\* — Standardized version of `H` used as autoencoder input



\---



\## Dataset Generation



\### Requirements



\- MATLAB R2023a or higher

\- 5G Toolbox

\- Deep Learning Toolbox



\### Como gerar localmente



If you prefer to reproduce the dataset from scratch:

```matlab

% Check the script:

generateDataset

```



The script executes the following steps:



1\. \*\*CDL-C Simulation\*\* — configures and simulates the MIMO channel with the 5G Toolbox

2\. \*\*H-Matrix Extraction\*\* — collects channel estimation over time/frequency

3\. \*\*Normalization\*\* — pre-processes `H` for the autoencoder input 

4\. \*\*Export\*\* — saves matrices in `.mat` format to the output directory

> ⚠️.



\---

\## Pre-Processing
Autoencoders perform better without redundancies. The pre-processing remove H matrix redundancies to optimize the model results. The pre-processing comprises 5 steps:

1\. Average over OFDM Symbols: It is assumed that the channel does not change drastically within the same slot and we calculated the average channel estimate over the 14 OFDM symbols changing the channel estimate matrix size from `\[624 14 Nrx × Ntx]` to `\[624 1 Nrx × Ntx]`

2\. 2D Discrete Fourier Transform (2D DFT): Two Dimensional (2D) Discrete Fourier Transform (DFT) is performed over subcarriers and Tx antennas for each Rx antenna and slot to transform the channel information from the frequency-spatial domain to the time (delay-angle) one. In frequency domain, channel information is spread out while it is concentrated in few points in the time domain. 2D DFT transform the dense original measured matrix in a sparse matrix (2D DFT).

3\. Truncate Delay: The matrix obtained from 2D DFT is truncated using a truncation factor to remove values that do not carry information. The truncation reduced subcarriers from 624 to 28.

4\. 2D Inverse Discrete Fourier Transform (2D IDFT): 2D IDFT is applied to return to frequency domain

5\. Complex to Real-Imaginary: Since autoencoders perform better with real value, the complex matrix element are split in real and imaginary parts

The pre-processing is resumed in the figure below:

<img width="1120" height="420" alt="8Tx4Rx_1" src="https://github.com/user-attachments/assets/68ac5810-c344-4964-b6a8-c5c7e8ad6f02" />




\## Citação



If this dataset or code is useful for your research, cite:

```bibtex

@misc{csi-dataset-5gnr,

&#x20; author = {Abdel Chabi},

&#x20; title  = {CSI Channel Dataset for 5G NR Autoencoder Feedback Compression},

&#x20; year   = {2026},

&#x20; url    = {https://github.com/Abdel83/csi-dataset-5gnr}

}

```



\---





