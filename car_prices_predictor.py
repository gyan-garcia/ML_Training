# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:15:55 2017

@author: ggarcia
"""

import os
from sklearn import linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg


# Just a helper function to predict and evaluate the results
def predict_and_plot(model, test_set, plot_title):

    # Let's prepare the data into something more readable    
    test_set_elements_count = test_set.shape[0]
    test_features = np.float64(test_set[0:test_set_elements_count, 0:3])
    real_prices = np.float64(test_set[0:test_set_elements_count, 3])
    car_names = test_set[0:test_set_elements_count, 4]
    
    # Do the prediction
    predicted_prices = model.predict(test_features)

    # Compare the predicted price vs the real price using a graph
    x_ind = np.arange(test_set_elements_count)
    fig = plt.figure()
    plt.scatter(x_ind, predicted_prices,   label="predicted price")
    plt.scatter(x_ind, real_prices,  label="real price")
    plt.xticks(x_ind, car_names, rotation='vertical')
    plt.legend(loc='best')
    fig.suptitle(plot_title, fontsize=20)
    plt.show()
    
    # Get the regression score
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    print ("R^2 Regression score:", metrics.r2_score(real_prices, predicted_prices))




   
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step 1: Collect Training Data, and arrange it
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Don't forget to set the path to whereever your data folder is.
os.chdir('C:\git\ML_Training') 

# https://www.cars.com/research/compare/
df = pd.read_csv('.\\car_prices.csv')

# Let's look at the data
#df.shape
#df.columns
df


# Let's get the features and prices so we can train the model
features = df.as_matrix(columns= df.columns[0:3])
prices = df.as_matrix(columns= df.columns[3:4])

#features
#prices




# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step 2: Train Model
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         
logistic_regression_model = linear_model.LogisticRegression()
logistic_regression_model.fit(features, prices.ravel())





# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step 3: Evaluate Model with some test data
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# https://www.cars.com/research/compare/?acodes=USC60KIC052A0,USC60FOC071A0,USC60BUC081A0
#                     MPG, HP,   Rating,  Price,  Name
test_set = np.array([[28,  185,  4.9,     22990,  "Kia Optima"],
                     [21,  240,  4.8,     28095,  "Ford Taurus"],
                     [23,  304,  4.3,     31990,  "Buick Lacrosse"],
                     [30,  240,  5.0,     34150,  "Volvo S60"],
                     [28,  150,  4.9,     22610,  "Volkswagen Jetta"],
                     [35,  106,  4.0,     15700,  "Scion iA"],])

predict_and_plot(logistic_regression_model, test_set, "Logistic regression")





# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Repeat 2/3: Train Model & Re-evalute 
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(features, prices.ravel())

passive_aggressive_model = linear_model.PassiveAggressiveRegressor() # Must be a Seattle thing.
passive_aggressive_model.fit(features, prices.ravel())

predict_and_plot(logistic_regression_model, test_set, "Logistic regression")
predict_and_plot(linear_regression_model, test_set, "Linear regression")
predict_and_plot(passive_aggressive_model, test_set, "Passive aggressive regression (best name ever!)")




# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step 4 : Enjoy!
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
img=mpimg.imread('the_homer.png')
imgplot = plt.imshow(img)
plt.show()

print ("How much the Homer mobile would cost? 82000 According to http://simpsons.wikia.com/wiki/The_Homer")
print ("Predicted price:", linear_regression_model.predict([[11,  510,  2.0]])[0])
print ("Predicted price:", passive_aggressive_model.predict([[11,  510,  2.0]])[0])
