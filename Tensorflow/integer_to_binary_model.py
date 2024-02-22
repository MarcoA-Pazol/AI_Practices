import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

"""THIS IS NOT WORKING CORRECTMENT, I NEED TO FOUND THE IDEAL MODEL, IT HAVE NOT BE SEQUENTIAL MODEL."""

#Setting data
integers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 20, 57, 100, 500, 539, 1000])
binaries = np.array([0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011, 10000, 10100, 111001, 1100100, 111110100, 1000011011, 1111101000])

#Creating model
layout = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layout])

#Compiling model
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#Training model
print('Starting model training')
historial = model.fit(integers, binaries, epochs=1000, verbose=True)

#Ploting result of the losing function during the model training
plt.title('Model training losing function magnitude')
plt.xlabel('# Epoch')
plt.ylabel('Loss magnitude')
plt.plot(historial.history['loss'])
plt.show

#Prediction
print('Comprobe the trained model doing a prediction:')
integer = 28
result = model.predict([28])
print(f'The result of {integer} is {result} according to your trained model.')

#Review internal model variables
print('Internal model variables:')
print(layout.get_weights())
