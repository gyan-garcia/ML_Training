# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:18:37 2017

@author: ggarcia
"""

import os
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_classification_model_decision_boundary(model, features, labels):
    xx, yy = np.mgrid[0:850:5, 0:850:5]
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    predicted_grid_probabilities = model.predict_proba(grid)
    
    f, ax = plt.subplots(figsize=(8, 6))
    #http://matplotlib.org/examples/color/colormaps_reference.html
    contour = ax.contourf(xx, yy, predicted_grid_probabilities.reshape(xx.shape[0],yy.shape[0]), 25, cmap="coolwarm",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    
    x_coords = features[0:features.shape[0], 0:1]
    y_coords = features[0:features.shape[0], 1:2]
    
    ax.scatter(x_coords, y_coords, c=labels, s=50, cmap="coolwarm", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)
    plt.show()



def plot_dataset(features, labels):
    x_coords = features[0:features.shape[0], 0:1]
    y_coords = features[0:features.shape[0], 1:2]
    
    f, ax = plt.subplots(figsize=(8, 6))    
    ax.scatter(x_coords, y_coords, c=labels, s=50, cmap="coolwarm", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)   
    plt.show()



def train_model(features, labels, hidden_activation, output_activation, loss_function):   
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=2, activation=hidden_activation)) 
    model.add(Dense(8, activation=hidden_activation))
    model.add(Dense(1, activation=output_activation))
    
    # Compile model
    model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy']) #loss='binary_crossentropy' # optimizer='adam'
    # loss function should be mse for linear
    # and binary_crossentropy for sigmoid (binary classification problem)
        
    # Fit the model
    model.fit(features, labels, epochs=300, batch_size=10)
    return model
    


# fix random seed for reproducibility
np.random.seed(7)
os.chdir('C:\git\ML_Training') 



# load linear dataset
df = pd.read_csv('.\\regression_data.csv')
features = np.array(df.as_matrix(columns= df.columns[0:2]))
labels = np.array(df.as_matrix(columns= df.columns[2:3]))
    
# Let's take a look into our dataset
plot_dataset(features, labels)

# Train the classifier and plot the classification decision boundary
model = train_model(features, labels, hidden_activation='linear', output_activation='linear', loss_function='mse')
plot_classification_model_decision_boundary(model, features, labels)



# load non-linear dataset
df = pd.read_csv('.\\regression_non_linear_data.csv')
features = np.array(df.as_matrix(columns= df.columns[0:2]))
labels = np.array(df.as_matrix(columns= df.columns[2:3]))
    
# Let's take a look into our dataset
plot_dataset(features, labels)

# Let's train a linear classifier to see how it behaves, 
model = train_model(features, labels, hidden_activation='linear', output_activation='linear', loss_function='mse')
plot_classification_model_decision_boundary(model, features, labels)

# Let's trains a non linear classifier.
model = train_model(features, labels, hidden_activation='relu', output_activation='sigmoid', loss_function='binary_crossentropy')
plot_classification_model_decision_boundary(model, features, labels)


# For more information of what loss funtions and optimizers to use: https://keras.io/getting-started/sequential-model-guide/#compilation

# For a multi-class classification problem
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

# For a binary classification problem
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

# For a mean squared error regression problem
#model.compile(optimizer='rmsprop',
#              loss='mse')





