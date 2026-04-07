#Import Pandas
import pandas as pd

#Import CSV File
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

#Define the features (X) and the target variable (y)
X = df.drop("medv", axis=1)
y = df["medv"]

#Import necessary libraries for data preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create an instance of the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#Make predictions on the test set
print(model.score(X_test_scaled, y_test))

#Feature Importance: Display the coefficients of the features to understand their impact on the target variable
print(pd.DataFrame({
    "Feature": X.columns,
    "Weight": model.coef_
}))

#Calculate the predicted values for the entire dataset (for visualization purposes)
y_pred = model.predict(X_test_scaled)

#Visualize the relationship between the true prices and the predicted prices
import matplotlib.pyplot as plt

#Plot a line representing perfect predictions (where predicted values equal actual values)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")

#Scatter plot of actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()