# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:15:55 2017

@author: ggarcia
"""

from sklearn import linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt

# Just a helper function to predict and evaluate the results
def predict_and_plot(model, test_set, plot_title):
    
    test_set_elements_count = test_set.shape[0]

    test_features = np.float64(test_set[0:test_set_elements_count, 0:3])
    real_prices = np.float64(test_set[0:test_set_elements_count, 3])
    car_names = test_set[0:test_set_elements_count, 4]
    
    predicted_prices = model.predict(test_features)
    
    x_ind = np.arange(test_set_elements_count)
    
    fig = plt.figure()
    plt.scatter(x_ind, predicted_prices,   label="predicted price")
    plt.scatter(x_ind, real_prices,  label="real price")
    plt.xticks(x_ind, car_names, rotation='vertical')
    plt.legend(loc='best')
    
    fig.suptitle(plot_title, fontsize=20)
    plt.show()
    
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    print ("R^2 (coefficient of determination) regression score:", metrics.r2_score(real_prices, predicted_prices))

    

# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step 1: Collect Training Data
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# https://www.cars.com/research/compare/
#                    MPG , HP,   Rating,  Price
data_set = np.array([[30,  138,  4.7,     15220],  # Sonic
                     [32,  120,  4.7,     15455],  # Fiesta
                     [30,  109,  4.2,     12855],  # Versa
                     [31,  138,  4.8,     15015],  # Rio
                     [31,  158,  4.7,     19515],  # Civic
                     [31,  145,  4.4,     18085],  # Elantra
                     [31,  132,  4.5,     18185],  # Corolla
                     [29,  138,  4.8,     16995],  # Cruze
                     [25,  175,  4.9,     23485],  # Fussion
                     [27,  185,  4.8,     23080],  # Accord
                     [31,  182,  4.8,     23365],  # Altima
                     [28,  178,  4.6,     23955],  # Camry
                     [31,  184,  4.7,     22330],  # Mazda 6
                     [23,  292,  4.8,     29090],  # Charger
                     [24,  268,  4.9,     33535],  # Avalon
                     [23,  292,  4.7,     33355],  # 300
                     [22,  293,  4.6,     33840],]) # Cadenza

data_set_elements_count = data_set.shape[0]

features = data_set[0:data_set_elements_count, 0:3]
prices = data_set[0:data_set_elements_count, 3]



# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step 2: Train Model
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         
logistic_regression_model = linear_model.LogisticRegression()
logistic_regression_model.fit(features, prices)



# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step 3: Evaluate Model
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
linear_regression_model.fit(features, prices)

# Must be a Seattle thing.
passive_aggressive_model = linear_model.PassiveAggressiveRegressor()
passive_aggressive_model.fit(features, prices)

predict_and_plot(logistic_regression_model, test_set, "Logistic regression")
predict_and_plot(linear_regression_model, test_set, "Linear regression")
predict_and_plot(passive_aggressive_model, test_set, "Passive aggressive regression")


# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step 4 : Enjoy!
# # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
print ("How much the Homer mobile would cost? 82000 According to http://simpsons.wikia.com/wiki/The_Homer")
print ("Predicted price:", linear_regression_model.predict([[11,  510,  2.0]])[0])
print ("Predicted price:", passive_aggressive_model.predict([[11,  510,  2.0]])[0])
