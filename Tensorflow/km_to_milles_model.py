import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np

#Setting data
kilometers = np.array([1, 2, 3, 7, 10, 16, 20, 58, 100, 150])
milles = np.array([0.621371, 1.24274, 1.86411, 4.3496, 6.21371, 9.94194, 12.4274, 36.0395, 62.1371, 93.2057])

data_dict = {
    'Kilometers':kilometers,
    'Milles':milles
}

data_frame = pd.DataFrame(data_dict)


#Data visualization
"""plt.title('Kilometer / Mille Comparative')
plt.xlabel('Kilometer')
plt.ylabel('Mille')
sns.scatterplot(x='Kilometers', y='Milles', data=data_frame)
plt.show()"""

#Loading data set
x_train = kilometers
y_train = milles

#Creating AI model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

model.summary()

#Compiling model
model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')

#Training model
epochs_history = model.fit(x_train, y_train, epochs=100)

#Evaluate trained model
print(epochs_history.history.keys())

#Plotting Lossing Progress

plt.plot(epochs_history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Lossing progress during model training')
plt.show()

#Get Weights
print(model.get_weights())

#Prediction using model
kilometers_imputed = 125
milles_returned = model.predict([kilometers_imputed])
print(f'Result using model prediction: {milles_returned}')
#Result using formula
kilometers_to_milles = kilometers_imputed * 0.621371
print(f'Result using Km - Milles formula: {kilometers_to_milles}')





