import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


#read Data
dataframe = pd.read_fwf("D:\ML\Siraj Raval\linear_regression_demo\\brain_body.txt")
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model

b = linear_model.LinearRegression()
b.fit(x_values,y_values)

#visualize data
plt.scatter(x_values,y_values)
plt.plot(x_values,b.predict(x_values))
plt.show()


