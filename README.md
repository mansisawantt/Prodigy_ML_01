# Prodigy_ML_01
---

# House Prices - Advanced Regression Techniques

## Project Aim

This project aims to build a linear regression model to predict house prices based on key features, aiding in data-driven decision-making in the real estate market.

## Table of Contents

1. [Project Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Analysis](#analysis)
6. [Results](#results)
7. [Contributing](#contributing)
8. [Acknowledgements](#acknowledgements)

## Dataset

The dataset used in this project is available on Kaggle:  
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

## Installation

To set up the environment, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Load the Dataset**:
   Import the dataset into a Pandas DataFrame and inspect its structure.

2. **Exploratory Data Analysis (EDA)**:
   - Analyze missing values and data distribution.
   - Identify key features affecting house prices.

3. **Data Preprocessing**:
   - Handle missing values and outliers.
   - Apply feature scaling and encoding where necessary.

4. **Model Building & Evaluation**:
   - Train a linear regression model.
   - Use GridSearchCV for hyperparameter tuning.
   - Evaluate performance using Mean Squared Error (MSE) and R² Score.

### Example Usage

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

Ensure that the dataset file (`house_prices.csv`) is in the correct directory path for loading.

## Analysis

### Exploratory Data Analysis
- Visualize data distributions and correlations.
- Detect missing values and apply necessary imputations.

### Feature Engineering & Selection
- Extract important features impacting house prices.
- Perform feature scaling and encoding where applicable.

### Model Training & Evaluation
- Train a Linear Regression model and tune hyperparameters using GridSearchCV.
- Evaluate model performance using MSE and R² Score.

## Results

Key insights from the analysis:
- **Feature Importance**: Identified key variables influencing house prices.
- **Model Performance**: Achieved an optimized regression model with minimal error.
- **Visualization**: Displayed actual vs predicted house prices to assess model accuracy.

## Acknowledgements

Thanks to the following libraries and tools used in this project:
- [Pandas](https://pandas.pydata.org/) - Data manipulation.
- [NumPy](https://numpy.org/) - Numerical computing.
- [Matplotlib](https://matplotlib.org/) - Data visualization.
- [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization.
- [Scikit-learn](https://scikit-learn.org/) - Machine learning models and evaluation.

---

