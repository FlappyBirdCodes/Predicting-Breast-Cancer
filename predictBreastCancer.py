import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Loading csv file
cancer_data = pd.read_csv("Breast_cancer_data.csv")

# Separating data based inputs and outputs
x = cancer_data[["mean_texture",
                 "mean_perimeter", "mean_area", "mean_smoothness"]]
y = cancer_data[["diagnosis"]]

# Splitting data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=6)

# Training linear regression model
lm = LinearRegression()
model = lm.fit(x_train, y_train)

y_predict = lm.predict(x_test)

print(lm.score(x_train, y_train))
print(lm.score(x_test, y_test))
