import pandas as pd
from sklearn.linear_model import LogisticRegression

# Loading csv file
cancer_data = pd.read_csv("Breast_cancer_data.csv")

# Separating data based inputs and outputs
x = cancer_data[["mean_radius", "mean_texture",
                 "mean_perimeter", "mean_area", "mean_smoothness"]]
y = cancer_data[["diagnosis"]]

# Training logistic regression model
model = LogisticRegression()
model.fit(x, y)

sample_input = [[12, 13, 124, 998, 0.1243]]
cancer_predictions = model.predict(sample_input)

print(cancer_predictions)