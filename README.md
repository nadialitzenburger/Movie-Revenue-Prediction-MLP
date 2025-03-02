# Movie-Revenue-Prediction-MLP

# Film Revenue Prediction Model

## Project Overview
This project implements a deep learning model to predict movie box office revenue based on movie titles and other numerical features. Using a dataset of movies from TMDB, the model analyzes both textual data (movie titles) and numerical metrics (runtime, vote average, and vote count) to estimate financial performance.

## Key Features
- Processes and combines text and numerical data in a single neural network
- Utilizes word embeddings to capture semantic meaning of movie titles
- Normalizes numerical features for improved model performance
- Implements a multi-layer neural network with customizable architecture

## Technologies Used
- **PyTorch**: Main framework for building and training the neural network
- **scikit-learn**: Used for data preprocessing and evaluation metrics
- **NumPy**: Data manipulation and numerical operations
- **Pandas**: Data loading and manipulation (implied in the workflow)
- **Python**: Core programming language

## Model Architecture
The model employs a hybrid neural network structure that processes both text and numerical data:

1. **Text Processing**:
   - An embedding layer converts movie titles into dense vector representations
   - Word embeddings are aggregated to form a sentence-level representation

2. **Numerical Features Processing**:
   - A dedicated linear layer processes normalized numerical features (runtime, vote average, vote count)
   - Z-score normalization applied to ensure consistent scale across features

3. **Combined Processing**:
   - Text and numerical features are concatenated into a unified representation
   - Three fully-connected layers with ReLU activations process the combined data
   - Final linear layer produces the revenue prediction

## Results
- **R² Score**: 0.646636 on the test set
- The model successfully captures the relationships between movie metadata and financial performance
- Training convergence was achieved within 25 epochs using Adam optimizer

## Dataset
The model was trained on the TMDB movie dataset, which includes:
- Movie titles
- Runtime information
- Vote averages
- Vote counts
- Revenue figures (target variable)

## Implementation Details
- Custom dataset class for efficient data loading and preprocessing
- Padding mechanism to handle variable-length text input
- Early stopping implementation to prevent overfitting
- Gradient clipping for stable training
- Learning rate of 0.020011 produced the best results
