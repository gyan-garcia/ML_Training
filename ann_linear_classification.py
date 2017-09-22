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

# fix random seed for reproducibility
np.random.seed(7)


os.chdir('C:\git\ML_Training') 


# load dataframe
df = pd.read_csv('.\\regression_data.csv')

X = np.array(df.as_matrix(columns= df.columns[0:2]))
Y = np.array(df.as_matrix(columns= df.columns[2:3]))


# create model
model = Sequential()
#model.add(Dense(12, input_dim=2, activation='relu', kernel_initializer='uniform'))
model.add(Dense(4, input_dim=2, activation='linear')) #relu
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear')) #sigmoid


# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #loss='binary_crossentropy' # optimizer='adam'

# Fit the model
model.fit(X, Y, epochs=500, batch_size=10)

# Normaly when the activation is linear, the loss should be MSE
# When the activation is sigmoid, the loss should be binary_crossentropy for a binary classification problem








xx, yy = np.mgrid[0:850:5, 0:850:5]

grid = np.c_[xx.ravel(), yy.ravel()]
predicted_grid_probabilities = model.predict_proba(grid)

f, ax = plt.subplots(figsize=(8, 6))
#http://matplotlib.org/examples/color/colormaps_reference.html
contour = ax.contourf(xx, yy, predicted_grid_probabilities.reshape(xx.shape[0],xx.shape[0]), 25, cmap="coolwarm",
                      vmin=0, vmax=1)
ax_c = f.colorbar(contour)

x_coords = X[0:X.shape[0], 0:1]
y_coords = X[0:X.shape[0], 1:2]

ax.scatter(x_coords, y_coords, c=Y, s=50,
           cmap="coolwarm", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
















