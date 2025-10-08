# House_Price_Prediction_model
# 🏠 USA Housing Price Prediction using Linear Regression

This project predicts **house prices** in the USA based on features like average area income, number of rooms, and population.  
It uses **Linear Regression** (a supervised machine learning model) to learn relationships between these features and housing prices.

---

## 📘 Project Overview

In this project, we:
1. Load and explore the **USA_Housing.csv** dataset.
2. Split the data into **training** and **testing** sets.
3. Apply **StandardScaler** to normalize the feature values.
4. Train a **Linear Regression model** using `scikit-learn`.
5. Evaluate the model by predicting prices on the test set.
6. Visualize **Actual vs Predicted** house prices.
7. Predict the price of a **new house** based on given input values.

---

## 🧠 Concepts Used

- **Machine Learning Type:** Supervised Learning  
- **Algorithm:** Linear Regression  
- **Libraries:**  
  - `numpy` — numerical operations  
  - `pandas` — data manipulation  
  - `matplotlib` — data visualization  
  - `scikit-learn` — ML model building and scaling  

---

## 📂 Dataset

The dataset used is `USA_Housing.csv`, which contains the following columns:

| Feature | Description |
|----------|--------------|
| `Avg. Area Income` | Average income of residents in the area |
| `Avg. Area House Age` | Average age of houses in the area |
| `Avg. Area Number of Rooms` | Average number of rooms per house |
| `Avg. Area Number of Bedrooms` | Average number of bedrooms per house |
| `Area Population` | Population of the area |
| `Price` | Target column — the house price |

---

## ⚙️ How It Works

1. Load the dataset
2. Split into training & testing sets
3. Scale the features
4. Train a Linear Regression model
5. Predict prices
6. Visualize the results

This workflow demonstrates a complete end-to-end machine learning project using Linear Regression for predicting house prices.


--- 
   
