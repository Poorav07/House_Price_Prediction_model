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

### 1. **Data Loading**
We start by loading the housing dataset using **pandas**:

```python
df = pd.read_csv("USA_Housing.csv")
print(df.head())


This gives a quick look at the first few rows of the dataset.

2. Feature Selection

We select the relevant features (independent variables) and the target column (Price):

X = df[['Avg. Area Income',
        'Avg. Area House Age',
        'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms',
        'Area Population']]
y = df['Price']

3. Splitting Data into Training and Testing Sets

To evaluate how well our model performs on unseen data, we split the dataset into 80% training and 20% testing:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

4. Feature Scaling

Since the features have different scales (like income vs population), we use StandardScaler to normalize them.
This helps the linear regression model perform better.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

5. Model Training

We train a Linear Regression model using the scaled data:

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

6. Model Prediction

Once trained, the model can predict housing prices on the test data:

y_pred = model.predict(X_test_scaled)
print(y_pred[:5])


This displays the first five predicted prices.

7. Predicting a New House

We can also predict the price of a new house by giving the model custom input values:

new_house = pd.DataFrame([[70000, 5, 6, 3, 30000]],
                         columns=['Avg. Area Income',
                                  'Avg. Area House Age',
                                  'Avg. Area Number of Rooms',
                                  'Avg. Area Number of Bedrooms',
                                  'Area Population'])

new_house_scaled = scaler.transform(new_house)
prediction = model.predict(new_house_scaled)
print(f"Predicted price: {prediction[0]:.2f}")

8. Visualization

To visually evaluate how close the predictions are to the actual prices, we plot a scatter plot:

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot(y_test, y_test, color='red', label='Perfect Prediction Line')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()


The red line represents perfect predictions.
The blue dots show how close the model's predictions are to the real values.

✅ In short:

Load the dataset

Split into training & testing sets

Scale the features

Train a Linear Regression model

Predict prices

Visualize the results

This workflow demonstrates a complete end-to-end machine learning project using Linear Regression for predicting house prices.


--- 
   
