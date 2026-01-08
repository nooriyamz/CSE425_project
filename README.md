# Unsupervised Learning Project: VAE for Hybrid Language Music Clustering

**Course:** Neural Networks  
**Prepared By:** Moin Mostakim

---

## Project Overview

This project implements an unsupervised learning pipeline inspired by Variational Autoencoders (VAE) for clustering hybrid language music tracks.  
The goal is to extract latent representations from audio and/or lyrics and perform clustering across easy, medium, and hard tasks.

---

## Repository Structure

### `data/` Folder

This folder contains the datasets used for the project.

- `audio/` – Audio files for music tracks  
  [Google Drive link for audio files](https://drive.google.com/drive/folders/your-audio-folder-link)

- `metadata/` – Metadata and lyric files for the tracks  
  [Google Drive link for metadata files](https://drive.google.com/drive/folders/your-metadata-folder-link)

Each subfolder includes a `README.md` describing its contents.

---

### `notebooks/` Folder

Contains Jupyter notebooks for exploratory analysis, organized by task difficulty:

- **Easy Task:** Basic VAE feature extraction and clustering  
- **Medium Task:** Convolutional VAE with hybrid audio + lyrics features  
- **Hard Task:** Conditional VAE / Beta-VAE for multi-modal clustering

[Open the Colab notebook here](https://colab.research.google.com/drive/1HUxMDML25a68BsSPNeI1fu0UR4ccTJ71?usp=sharing)

The folder also includes a `README.md` explaining the notebooks and their respective tasks.

---

### `src/` Folder

Contains Python scripts for the main pipeline:

- `vae.py` – Variational Autoencoder implementation  
- `dataset.py` – Dataset loading and preprocessing  
- `clustering.py` – Clustering algorithms  
- `evaluation.py` – Metric computation and evaluation  

---

### `results/` Folder

- `latent_visualization/` – Visualizations of the latent space  
- `clustering_metrics.csv` – Clustering performance metrics for different experiments  

---

### Other Files

- `requirements.txt` – Python dependencies  
- `README.md` – This file  

---

## Notes

This repository is organized to support reproducibility and easy navigation for all project tasks (easy, medium, hard). All datasets, notebooks, and results are linked and structured for clarity.
