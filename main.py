import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD DATA
print("Step 1: Loading data...")
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. SELECT FEATURES
# We use 'MedInc' (Income) as our main predictor for simplicity
X = df[['MedInc', 'HouseAge', 'AveRooms']] 
y = df['MedHouseVal']

# 3. SPLIT DATA (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE MODEL
model = LinearRegression()
model.fit(X_train, y_train)
print("Step 2: Model trained successfully!")

# 5. EVALUATE
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"\n--- Results ---")
print(f"Accuracy (R2 Score): {r2:.2f}")
print(f"Error (RMSE): {rmse:.2f}")

# 6. SHOW AN EXAMPLE PREDICTION
# Let's predict the price for the first house in our test set
sample_prediction = model.predict(X_test.head(1))
print(f"\nExample Prediction: ${sample_prediction[0]*100000:.2f}")
print(f"Actual Price: ${y_test.iloc[0]*100000:.2f}")

# Create a small table to show the importance of each feature
importance = pd.DataFrame(model.coef_, X.columns, columns=['Impact on Price'])
print("\n--- Feature Importance ---")
print(importance)

import matplotlib.pyplot as plt

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Create a small table to show the 'weight' of each feature
importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\n--- Feature Importance (Coefficients) ---")
print(importance)