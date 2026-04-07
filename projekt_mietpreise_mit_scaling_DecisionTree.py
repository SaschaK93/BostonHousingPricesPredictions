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

#Train a Decision Tree Regressor model
from sklearn.tree import DecisionTreeRegressor

#Initialize the Decision Tree Regressor model with a random state for reproducibility
#model = DecisionTreeRegressor(random_state=42) / training score was 1.0, but test score was 0.85, which indicates overfitting. 
# To reduce overfitting, we can limit the depth of the tree by setting the max_depth parameter. 
# A common choice is to set max_depth to a value between 3 and 5, depending on the complexity of the dataset.
model = DecisionTreeRegressor(max_depth=5, random_state=42)

#Fit the model to the training data
model.fit(X_train_scaled, y_train)

#Evaluate the model's performance on the test set
print("Test:",model.score(X_test_scaled, y_test))

#Evaluate the model's performance on the training set
print("Train:",model.score(X_train_scaled, y_train))

#Visualize the Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Set the figure size for better visualization
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    max_depth=3   # wichtig für Übersicht!
)

plt.show()