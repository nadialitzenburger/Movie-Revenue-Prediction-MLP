# Film Revenue Prediction Model: Comparing Pre-Release vs. Post-Release Features

## Research Overview
This project investigates how accurately movie box office revenue can be predicted using neural networks with different feature sets. The key research question is whether a model using only pre-release data can achieve comparable accuracy to one that uses post-release data. This has practical implications for film studios needing to make financial decisions before a movie is released.

## Models Comparison
Two different models are implemented and compared:

1. **Baseline Model** (`baseline_model_notebook.ipynb`, `baseline_model_py.py`):
   - Uses post-release features: movie title, runtime, vote average, vote count
   - Achieved an R² score of 0.647 on the test set
   - Better accuracy but less practical for pre-production decisions

2. **Pre-Release Model** (`comparison_model_notebook.ipynb`, `comparison_model_py.py`):
   - Uses only features available before release: movie title, runtime, release month
   - Achieved an R² score of 0.212 on the test set
   - Lower accuracy but more useful for advance planning

## Key Findings
- Post-release features (especially vote average and vote count) significantly improve prediction accuracy (43% higher R² score)
- Runtime is the most influential pre-release feature with a correlation coefficient of 0.212
- Even with limited pre-release data, the model performs significantly better than random guessing
- Model architecture differences have less impact on performance than feature selection

## Technical Implementation
- **Framework**: PyTorch implementation of multilayer perceptrons (MLPs)
- **Text Processing**: Embedding layer for movie titles
- **Numerical Features**: Z-score normalization
- **Missing Values**: Zero-imputation in baseline model, median-imputation in pre-release model
- **Optimization**: Adam/AdamW optimizer with weight decay for regularization
- **Evaluation**: R² score as primary metric

## Research Context
This work contributes to the broader field of applied machine learning in the film industry by:
- Quantifying the accuracy gap between pre-release and post-release prediction models
- Identifying the most valuable features for revenue prediction
- Proposing a practical approach for early-stage revenue forecasting

The findings suggest that while perfect revenue prediction remains challenging with pre-release data alone, the developed model can still provide valuable insights for production companies making early financial decisions.
