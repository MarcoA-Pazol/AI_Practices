import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Setting data
celsius = np.array([-40, -10, 0, 8, 15, 22, 38])
farenheit = np.array([-40, 14, 32, 46, 59, 72, 100])

#Designing model
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

#Compiling model
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error"
)

#Training model
print('Training model')
historial = model.fit(celsius, farenheit, epochs=1000, verbose=False)
print('Model trained succesfully!')

#Ploting results of the lossing function
plt.title("Lossing function magnitude")
plt.xlabel("# Epoch")
plt.ylabel("Loss magnitude")
plt.plot(historial.history["loss"])
plt.show()

#Prediction
print('Doing prediction')
result = model.predict([100])
print(f'The result is {result} farenheit')

#Review internal model variables
print('Internal model variables:')
print(layer.get_weights())





