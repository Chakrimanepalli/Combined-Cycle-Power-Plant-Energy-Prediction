# Combined Cycle Power Plant Energy Prediction

## Overview
This project aims to predict the net hourly electrical energy output of a combined cycle power plant using machine learning models. The dataset contains environmental variables such as temperature, exhaust vacuum, ambient pressure, and relative humidity.

## Dataset
The dataset consists of 9568 observations collected over six years when the power plant was operating at full load. It includes the following features:
- Temperature (in degrees Celsius)
- Exhaust Vacuum (in cm Hg)
- Ambient Pressure (in millibar)
- Relative Humidity (in percentage)
- Energy Production (in MW)

## Models Used
The following machine learning models were employed for prediction:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Elastic Net Regression
- Random Forest Regression
- Decision Tree Regression
- XGBoost Regression
- Gradient Boosting Regression

## Evaluation
Model performance was evaluated using Mean Squared Error (MSE) and R-squared values.

## Results
Based on the evaluation metrics, the XGBoost Regression model demonstrated the best performance in predicting energy production.

## Usage
1. Clone the repository.
2. Install the required libraries listed in `requirements.txt`.
3. Run the Jupyter Notebook `combined_cycle_power_plant.ipynb` to see the data preprocessing, model training, evaluation, and results.

## Contributors
- Chakravarthi Manepalli

