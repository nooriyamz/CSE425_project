
---

## Data Folder

- **audio/** – Contains audio clips for music tracks. [Google Drive link](https://drive.google.com/drive/folders/1n2GlQ5uMvyyWxFbMrH0StO5epqhs57fe)
- **metadata/** – Contains CSV files with song metadata and lyrics. [Google Drive link](https://drive.google.com/file/d/1yKeQM1ADi4pz9Mb3aqEGtQYLmLWsAlxT/view?usp=sharing)

---

## Notebooks Folder

Contains Jupyter/Colab notebooks for the following tasks:

- **Easy Task:** Basic VAE feature extraction and PCA + KMeans clustering  
- **Medium Task:** Convolutional VAE with hybrid audio + lyrics embeddings, multiple clustering algorithms  
- **Hard Task:** Beta-VAE or Conditional VAE, multi-modal clustering with audio, lyrics, and genre features

[Colab Notebook Link](https://colab.research.google.com/drive/1HUxMDML25a68BsSPNeI1fu0UR4ccTJ71?usp=sharing)

---

## Results Folder

- **latent_visualization/** – Visualizations of the latent space (UMAP, t-SNE plots, reconstructions)  
- **clustering_metrics.csv** – Performance metrics for different experiments (Silhouette Score, CH Index, NMI, ARI, Purity)

---

## Source Code (`src/`)

- **vae.py** – VAE / Beta-VAE implementation  
- **dataset.py** – Dataset loading, preprocessing, and Mel-spectrogram extraction  
- **clustering.py** – KMeans, Agglomerative Clustering, DBSCAN implementations  
- **evaluation.py** – Metrics computation (Silhouette, CH, NMI, ARI, Purity)

---

## Usage

1. Clone the repository.  
2. Download the **audio** and **metadata** files from Google Drive.  
3. Install dependencies from `requirements.txt`.  
4. Open the Colab notebook in `notebooks/` to run Easy, Medium, or Hard tasks.  
5. Results and visualizations are stored in the `results/` folder.

---

## References / Datasets

- **Million Song Dataset (MSD):** http://millionsongdataset.com/  
- **GTZAN Genre Collection:** http://marsyas.info/downloads/datasets.html  
- **Jamendo Dataset:** https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset  
- **MIR-1K Dataset:** https://sites.google.com/site/unvoicedsoundseparation/mir-1k  
- **Lakh MIDI Dataset (LMD):** https://colinraffel.com/projects/lmd/  
- **Kaggle Lyrics Datasets:** https://www.kaggle.com/datasets?search=lyrics
