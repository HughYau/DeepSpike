# DeepSpike
Deep-Spike: Foundation Model-based  Pipeline for Large-Scale Spike Sorting of Neural Activity

## Overview

Spike sorting of high-resolution neural recordings is essential for understanding brain activity, but it remains challenging when multiple units are recorded due to their overlapping spike timing, low signal-to-noise ratios and overlapping clusters. Here, we introduce DeepSpike, a self-supervised deep learning model that automates spike sorting and overcomes key limitations of conventional spike sorting methods. DeepSpike is pretrained on large-scale unlabelled spiking events obtained from electrophysiological data as a general foundation model, enabling it to generalize to new recordings without dataset-specific retraining. DeepSpike uses a self-supervised autoencoder to learn robust low-dimensional spike embeddings that facilitate accurate clustering and effective noise filtering. The model is trained on a new, large-scale dataset consisting of $255M$ spiking events (SpikeVault-255M) derived from real *in vivo* recordings of about $4560$ minutes duration. The dataset consists of $15M$ ground truth spikes that are manually verified by an expert user. DeepSpike outperformed state-of-the-art spike sorting algorithms in both accuracy and robustness in our experiments on SpikeVault-255M, and two public benchmark datasets. Our results demonstrate that DeepSpike provides a scalable and generalizable solution for large-scale neural spike sorting. SpikeVault-255M dataset and the pretrained DeepSpike model are provided for further use and development.

## Features

- End-to-end spike sorting workflow
- Deep learning-based feature extraction (Autoencoder, VAE)
- Multiple clustering methods (GMM, DPGMM, HDBSCAN, KMeans)
- Integration with [SpikeInterface](https://spikeinterface.readthedocs.io/) for standardized spike sorting and evaluation
- Visualization tools for embeddings and clustering results
- Support for large public datasets

## Repository Structure

- `clustering.py`: Clustering algorithms and utilities
- `dataset.py`: Dataset loading and preprocessing
- `models.py`: Deep learning models (VAE)
- `preprocess.py`: Data preprocessing functions
- `utils.py`: Utility functions
- `models/`: Pretrained model weights
- `notebooks/`: Example Jupyter notebooks and analysis pipelines
- `tables/`: SpikeVault255M and Public dataset metrics and recording details

## Getting Started

1. **Clone the repository**
   ```sh
   git clone https://github.com/HughYau/DeepSpike.git
   cd DeepSpike
   ```

2. **Install dependencies**
   Make sure you have Python 3.8+ and install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

2. **Run example notebooks**
    Open notebooks/deep_spike_guideline.ipynb for a step-by-step demonstration.