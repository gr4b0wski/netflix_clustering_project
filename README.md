# Netflix Content Clustering & Recommendation Engine (GPU-Accelerated)

An advanced, end-to-end Machine Learning project applying unsupervised learning techniques and Natural Language Processing (NLP) to segment Netflix movies and TV shows into highly cohesive clusters. 

The primary business objective of this project is to build the foundational logic for a **Content-Based Recommendation System** that aligns with individual user preferences, ultimately driving up user retention and engagement on the streaming platform.

## Business Objectives
- **Increase User Retention:** By accurately suggesting content based on previously watched titles, keeping users engaged longer.
- **Content Portfolio Optimization:** Discovering underlying patterns in Netflix's catalog to identify saturated genres or underserved niches.

## Machine Learning Pipeline & Architecture

### 1. Data Ingestion & Reproducibility
To ensure 100% reproducibility, the project eliminates manual data handling. The dataset (Netflix catalog up to 2025) is fetched dynamically during runtime using the `kagglehub` API directly into the local cache.

### 2. Advanced NLP & Feature Engineering
- **Text Preprocessing:** Stopwords removal, stemming (SnowballStemmer), and tokenization using `nltk`.
- **Vectorization:** Transformation of unstructured text (descriptions, cast, genres) into high-dimensional numerical vectors using `TF-IDF`.
- **Deep Learning Embeddings:** Utilizing pre-trained **BERT Transformers** (HuggingFace) to capture deep semantic meaning from movie plots.

### 3. Dimensionality Reduction
High-dimensional text data suffers from the "curse of dimensionality". To optimize clustering algorithms, the feature space was reduced using:
- **PCA** (Principal Component Analysis)
- **UMAP** (Uniform Manifold Approximation and Projection) for superior non-linear topology preservation.

### 4. GPU-Accelerated Clustering Models
Multiple distance-based and density-based grouping algorithms were evaluated, including **K-Means, MiniBatchKMeans, Agglomerative, Birch, Gaussian Mixture, and DBSCAN**. 

To handle large-scale matrix operations efficiently, the project leverages **NVIDIA RAPIDS (cuML, cuDF) and CuPy** for seamless GPU acceleration, cutting down training time significantly compared to standard CPU execution.

Models were evaluated and optimized via `optuna` using metrics such as:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

## Tech Stack
- **Language:** Python 3.x
- **Hardware Acceleration:** CUDA, PyTorch, cuML, cuDF, CuPy (RAPIDS)
- **NLP & Transformers:** HuggingFace `transformers`, `nltk`, `scikit-learn`
- **Machine Learning:** `scikit-learn`, `umap-learn`, `optuna`
- **Data Visualization:** `matplotlib`, `seaborn`, `wordcloud`

## How to Run Locally

*Note: To fully utilize the RAPIDS libraries (`cuML`, `cuDF`), a CUDA-enabled NVIDIA GPU is required. The code includes a fallback mechanism if CUDA is not detected.*

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/netflix_clustering_project.git](https://github.com/yourusername/netflix_clustering_project.git)
   ```
2. Navigate to the directory and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `NetflixClusteringProject.ipynb` and run the cells sequentially. The raw dataset will be downloaded automatically.

