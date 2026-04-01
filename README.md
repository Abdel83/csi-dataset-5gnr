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



\- \*\*`H`\*\* — Complex channel matrices, pre-processed from  `\[624 14 Nrx × Ntx]` to `\[28 14 Nrx × Ntx]`

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





