
# Banknote Authentication using Support Vector Machines (SVM)

## Project Overview
The goal of this project is to predict whether a banknote is genuine or forged based on features such as Variance, Skewness, Curtosis, and Entropy. This is achieved using Support Vector Machines (SVM) with both Linear and Radial Basis Function (RBF) kernels. The project incorporates hyperparameter tuning and dimensionality reduction techniques to enhance model performance and visualization.

---

## Dataset Details
The dataset contains features that describe the properties of banknotes along with their corresponding labels (genuine or forged). The dataset is preprocessed and read into a DataFrame for analysis and manipulation.

### Sample Dataset
| Variance | Skewness | Curtosis | Entropy  | Target |
|----------|----------|----------|----------|--------|
| 3.62160  | 8.6661   | -2.8073  | -0.44699 | 0      |
| 4.54590  | 8.1674   | -2.4586  | -1.46210 | 0      |
| 3.86600  | -2.6383  | 1.9242   | 0.10645  | 0      |
| 3.45660  | 9.5228   | -4.0112  | -3.59440 | 0      |
| 0.32924  | -4.4552  | 4.5718   | -0.98880 | 0      |

### Dataset Insights
- Mean of each column:
  - Variance: 0.433735
  - Skewness: 1.922353
  - Curtosis: 1.397627
  - Entropy: -1.191657
  - Target: 0.444606
- Null values in the dataset: None

---

## Methodology

### Exploratory Data Analysis (EDA)
- Histograms, pair plots, and correlation heatmaps were created to analyze the data distribution, interrelationships between features, and potential patterns or anomalies.

### Data Preprocessing
- **Feature Scaling**: Standardization was applied to ensure all features contribute equally during model training.

### Splitting the Data
- The dataset was split into training (80%) and testing (20%) sets.

### Model Training
1. **Linear Kernel**: Trained to handle linearly separable data for simplicity.
2. **RBF Kernel**: Used for non-linear data patterns.

### Model Evaluation
- Metrics: Precision, recall, F1-score, and accuracy.
- Cross-validation was performed to ensure generalization.

### Hyperparameter Tuning
- Grid Search was used to optimize the hyperparameters (`C` and `gamma`) for the RBF kernel.
- Best Parameters:
  - C = 10
  - Gamma = 0.5
  - Kernel = 'rbf'

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reduced the data to two dimensions for visualization.
- The SVM decision boundary was visualized in 2D space after PCA.

---

## Key Results
- The SVM model with the RBF kernel achieved:
  - **100% accuracy** during cross-validation.
  - Best parameters: C = 10, gamma = 0.5.
- The RBF kernel outperformed the linear kernel in classification accuracy.
- The decision boundary and PCA visualizations provided intuitive insights into data classification.

---

## Conclusion
This project demonstrates the effectiveness of Support Vector Machines (SVM) for solving the Banknote Authentication problem. The combination of feature scaling, hyperparameter tuning, and dimensionality reduction resulted in a robust model capable of perfect classification accuracy. The results underscore the power of SVMs in handling complex classification tasks.

---

## File Structure
- **`dataset.csv`**: Contains the banknote features and labels.
- **`svm_banknote_authentication.py`**: Python script for data preprocessing, model training, evaluation, and visualization.
- **`README.md`**: Project documentation.

---

## Instructions for Running the Project
1. Clone the repository.
2. Install required libraries: `pip install -r requirements.txt`.
3. Run the Python script: `python svm_banknote_authentication.py`.
4. View the results and visualizations in the output.

---

## Technologies Used
- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

---

## Acknowledgements
- Dataset Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

---


