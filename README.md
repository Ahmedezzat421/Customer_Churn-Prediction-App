# Customer Churn Prediction

A machine learning and deep learning project for predicting whether a customer is likely to **churn** based on customer account and service information.

The project compares three traditional machine-learning classifiers—**Random Forest, SVM, and KNN**—with a **PyTorch MLP deep-learning model**, then provides a Gradio web interface for making predictions on new customers.

## Project Overview

The workflow includes:

1. Loading the customer churn dataset.
2. Removing the `CustomerID` identifier.
3. Checking for missing values.
4. Encoding categorical features.
5. Splitting the data into training and testing sets.
6. Standardizing numerical features.
7. Performing exploratory data visualization.
8. Training Random Forest, SVM, and KNN classifiers.
9. Building and training a PyTorch neural network.
10. Comparing model performance using Accuracy, Precision, Recall, and F1-Score.
11. Generating confusion matrices.
12. Predicting churn for new customer data.
13. Providing an interactive Gradio prediction interface.

## Dataset

The project expects a CSV file named:

```text
customer_churn.csv
```

The model uses the following input features:

- `Tenure` — customer tenure in months
- `MonthlyCharges` — monthly customer charges
- `TotalCharges` — total customer charges
- `Contract` — contract type
- `TechSupport` — whether technical support is available

The target variable is:

```text
Churn
```

`CustomerID` is removed because it is an identifier rather than a predictive feature.

## Data Preprocessing

Categorical variables are converted using one-hot encoding:

- `Contract`
- `TechSupport`

The encoded features are split into:

- **80% training data**
- **20% testing data**

The split uses `random_state=42` and stratification based on the target variable.

The numerical features are standardized using `StandardScaler`:

```text
Tenure
MonthlyCharges
TotalCharges
```

## Exploratory Data Analysis

The notebook generates visualizations for:

- Churn count by contract type
- Tenure distribution by churn status
- Monthly charges distribution by churn status
- Feature correlation heatmap

These plots help examine relationships between customer characteristics and churn.

## Machine Learning Models

Three traditional classification models are trained:

### Random Forest

```python
RandomForestClassifier(random_state=42)
```

### Support Vector Machine

```python
SVC(probability=True, random_state=42)
```

### K-Nearest Neighbors

```python
KNeighborsClassifier()
```

Each model is trained using the processed training data and evaluated on the test set.

## Deep Learning Model

The project also implements a binary-classification MLP using PyTorch.

Architecture:

```text
Input
  ↓
Linear(input_dim → 16)
  ↓
ReLU
  ↓
Linear(16 → 8)
  ↓
ReLU
  ↓
Linear(8 → 1)
  ↓
Sigmoid
```

The model uses:

- **Loss:** Binary Cross Entropy (`BCELoss`)
- **Optimizer:** Adam
- **Learning Rate:** `0.005`
- **Epochs:** `80`
- **Batch Size:** `32`

The training set is further divided to create a **15% validation subset**.

## Evaluation Metrics

The following metrics are calculated for every model:

- Accuracy
- Precision
- Recall
- F1-Score

The project also generates confusion matrices for:

- Random Forest
- SVM
- KNN
- Deep Learning (PyTorch)

A consolidated comparison chart is produced to compare the models.

> Note: The exact performance values depend on the contents of `customer_churn.csv` and the execution environment. The source code does not contain fixed final metric values.

## Prediction on New Customers

The project includes a reusable `predict_churn()` function.

Example input:

```python
sample_customer = {
    'Tenure': 6,
    'MonthlyCharges': 85.00,
    'TotalCharges': 510.00,
    'Contract': 'Month-to-month',
    'TechSupport': 'No'
}
```

The function returns:

- Predicted class: `Churn` or `No Churn`
- Churn probability as a percentage

A prediction threshold of **0.5** is used.

## Gradio Application

The project includes an interactive Gradio interface.

The user can enter:

- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Tech Support status

The application returns:

```text
Prediction: Churn / No Churn
Churn Probability: XX.XX%
```

The interface is configured to launch with:

```python
interface.launch(share=True)
```

## Requirements

Install the main Python dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch gradio
```

The original notebook also installs Gradio directly with:

```python
!pip install -q gradio
```

## How to Run

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-DIRECTORY>
```

### 2. Add the dataset

Place:

```text
customer_churn.csv
```

in the project working directory.

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch gradio
```

### 4. Run the notebook

The source was originally generated in **Google Colab**. Upload the notebook/code and dataset to Colab and execute the cells in order.

Alternatively, the Python code can be adapted to run locally.

## Project Structure

```text
.
├── customer_churn.csv
├── untitled33(1).py
└── README.md
```

## Reproducibility

The project uses fixed random seeds in several stages:

```python
random_state=42
```

and:

```python
torch.manual_seed(42)
```

This helps make the training and evaluation process more reproducible.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- PyTorch
- Gradio
- Google Colab

## Author

**Ahmed Ezzat**

## Project Goal

The main goal of this project is to build and compare different machine-learning approaches for customer churn prediction, while demonstrating a complete workflow from data preprocessing and exploratory analysis to model training, evaluation, inference, and deployment through an interactive interface.
