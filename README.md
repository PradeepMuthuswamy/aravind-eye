# Aravind Eye Care Regression Model

This Streamlit application builds regression models to analyze and predict values from the Aravind Eye Care dataset.

## Features

- **Data Exploration**: View and analyze the dataset with interactive visualizations
- **Regression Modeling**: Build Linear Regression or Random Forest Regression models
- **Model Evaluation**: Evaluate model performance with metrics like MSE, RMSE, and R²
- **Interactive Predictions**: Make predictions with custom input values

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Use the application:
   - Explore the dataset with visualizations
   - Select a target variable (TOT_QTY or PRICE)
   - Choose a regression model type
   - Select features for the model
   - Build and evaluate the model
   - Make predictions with custom inputs

## Data Description

The Aravind Eye Care dataset contains the following columns:

- **Obs**: Observation number
- **TOT_QTY**: Total quantity
- **PRICE**: Price of the product
- **Ln_Tot_Qty**: Log of total quantity
- **Ln_Price**: Log of price
- **YEAR**: Year of the observation
- **MONTH**: Month of the observation
- **STATE**: State in India
- **CHANNEL**: Distribution channel
- **PRODUCTS**: Product type

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly