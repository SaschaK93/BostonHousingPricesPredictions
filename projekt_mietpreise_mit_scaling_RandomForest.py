#Import Pandas
import pandas as pd

#Import CSV File
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

#Define the features (X) and the target variable (y)
X = df.drop("medv", axis=1)
y = df["medv"]

#Select a subset of features for the model based on domain knowledge or feature importance
features = ["rm", "lstat", "dis", "crim"]
X_reduced = df[features]

#Import necessary libraries for data preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

#Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Import the Random Forest Decision Tree Regressor model from scikit-learn
from sklearn.ensemble import RandomForestRegressor

#Initialize the Random Forest Regressor model with specific hyperparameters
model = RandomForestRegressor(
    n_estimators=200,#number of trees in the forest
    max_depth=10,#maximum depth of the tree
    min_samples_split=5,#minimum number of samples required to split an internal node
    min_samples_leaf=2,#minimum number of samples required to be at a leaf node
    random_state=42)#random seed for reproducibility

#Fit the model to the training data
model.fit(X_train_scaled, y_train)

#Evaluate the model's performance on the test set
print("Test:",model.score(X_test_scaled, y_test))

#Evaluate the model's performance on the training set
print("Train:",model.score(X_train_scaled, y_train))

#Calculate and print the feature importance of each feature in the model
importance = pd.DataFrame({"Feature": X_reduced.columns,"Wichtigkeit": model.feature_importances_}).sort_values(by="Wichtigkeit", ascending=False)
print(importance)