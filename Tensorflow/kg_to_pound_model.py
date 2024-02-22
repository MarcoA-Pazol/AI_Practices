import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf

#Setting Data
kilogrames = np.array([1, 2, 3, 5, 10, 17, 20, 50, 100, 1000])
pounds = np.array([2.20462, 4.40925, 6.61387, 11.0231, 22.0462, 37.4786, 44.0925, 110.231, 220.462, 2204.62])

data_dict = {
    'Kg':kilogrames,
    'Lb':pounds
}

data_frame = pd.DataFrame(data_dict)

#Ploting Data
"""plt.title('Kg / Lb comparative')
plt.xlabel('Kg')
plt.ylabel('Lb')
sns.scatterplot(x='Kg', y='Lb', data=data_frame)
plt.show()"""


#Loading dataset
x_train = kilogrames
y_train = pounds

#Creating Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
print(model.summary())

#Compiling model
model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')

#Training model
epochs_history = model.fit(x_train, y_train, epochs=250, verbose=True)

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

#Prediction Using model
kg_imputed = int(input('Introduce Kg:'))
lb_returned = model.predict([kg_imputed])
print(f'Result using model prediction: {lb_returned}')
#Result using formula
lb_returned = kg_imputed * 2.20462
print(f'Result using formula: {lb_returned}')

