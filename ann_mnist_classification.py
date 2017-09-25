# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:30:12 2017

@author: ggarcia
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense 
import matplotlib.pyplot as plt

# Function to display the one dimentional array into a 2 dimentional digit.
def display_digit(data, labels, i):
    img = data[i]
    plt.title('Example %d. Label: %d' % (i, labels[i]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)



# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Let's dispplay whatever number is on position 50
display_digit(x_train, y_train, 50)

# Let's dispplay whatever number is on position 550
display_digit(x_train, y_train, 550)


num_classes = 10


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# create model
model = Sequential()
model.add(Dense(16, input_dim=784, activation='relu')) 
model.add(Dense(8, activation='relu')) 
   
model.add(Dense(num_classes, input_dim=784, activation='softmax')) 

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

        
# Fit the model
model.fit(x_train.reshape(60000, 784), y_train, epochs=5, batch_size=32)

# Score the model
scores = model.evaluate(x_test.reshape(10000, 784), y_test, verbose=0)
print('test score:', scores[0])
print('test accuracy:', scores[1])
    


# Let's try to print the weights
layers = model.layers
weights = layers[0].get_weights()

f, axes = plt.subplots(2, 5, figsize=(10,4))
axes = axes.reshape(-1)
for i in range(len(axes)):
    a = axes[i]
    a.imshow((weights[0][0:784, i:i+1]).reshape(28, 28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(())
    a.set_yticks(())
plt.show()



 
 