# Sentiment Analysis with Machine Learning Models

## Overview
This project evaluates and compares the performance of supervised and unsupervised machine learning models in binary text classification tasks, specifically sentiment analysis. The focus is on three models:
- **Support Vector Machines (SVM)**: A powerful supervised learning method for high-dimensional spaces.
- **Multilayer Perceptrons (MLP)**: A type of neural network for capturing complex patterns.
- **K-Means Clustering**: An unsupervised learning method that groups data points without predefined labels.

The study assesses these models' performance using metrics such as accuracy, precision, recall, and ROC-AUC, and visualizations such as confusion matrices and learning curves.

## Motivation
Inspired by papers such as *Sentiment Analysis for Movie Reviews* and *Predicting Star Ratings of Movie Review Comments*, this project explores binary classification and focuses on models like SVM, Neural Networks, and K-Means. The findings aim to uncover the strengths and weaknesses of each method in the context of natural language processing (NLP) tasks.

## Dataset
The dataset used is the **IMDB Movie Reviews Dataset**, sourced from Kaggle. It contains 50,000 reviews evenly split into positive and negative sentiment labels. Key preprocessing steps include:
- Text cleaning: Lowercasing, removing punctuation, and stop words.
- TF-IDF Vectorization: Converting text into numerical features.
- Splitting: 80% training and 20% testing.

## Models
1. **Support Vector Machines (SVM)**
   - Implementation: `LinearSVC` and `SVC` from `scikit-learn`.
   - Features: Decision scores, probabilistic predictions.
2. **Multilayer Perceptrons (MLP)**
   - Implementation: `MLPClassifier` from `scikit-learn` with ReLU activation and the Adam solver.
3. **K-Means Clustering**
   - Implementation: `KMeans` from `scikit-learn`.
   - Method: Assigns clusters to labels based on the mode of assignments.

## Results
- **Accuracy**:
  - SVM: **89.42%**
  - MLP: **87.47%**
  - K-Means: **51.82%**
- **ROC-AUC**:
  - SVM: **0.96**
  - MLP: **0.94**
- SVM outperformed MLP in interpretability and slightly in accuracy.
- K-Means struggled due to the unsupervised nature and lack of labeled data.

## Visualizations
Key visualizations include:
- **Confusion Matrices**: Highlighted model-specific strengths and weaknesses.
- **ROC Curves**: Showed the trade-off between sensitivity and specificity.
- **Learning Curves**: Examined overfitting and generalization trends.

## Tools and Libraries
- Python
- `scikit-learn`
- `matplotlib`, `seaborn` for visualizations

## Future Directions
- Implement transformer-based models like BERT or GPT for better semantic understanding.
- Explore semi-supervised methods to leverage both labeled and unlabeled data.

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
