import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.callbacks import TensorBoard
import pandas as pd

"""Datasets"""
#Data arrays
pounds = []
kilogrames = np.array([1, 2, 5, 10, 17, 20, 50, 85, 100, 104, 177, 820, 1000, 1289, 5270])

for kg in kilogrames:
    pound = kg * 2.20462
    pounds.append(pound)
    
pounds = np.array(pounds)

#Data dict
data_dict = {
    "pound": pounds,
    "kilograme": kilogrames
}

#Data frame
data_frame = pd.DataFrame(data_dict)

#Loading dataset
x_train = kilogrames
y_train = pounds

"""Keras Model"""
#Architecture
my_model = keras.models.Sequential([
    keras.layers.Dense(units=1, input_shape = [1], activation="relu"),
    keras.layers.Dense(1, activation="linear")
])

#Compiling
my_model.compile(optimizer = "adam", loss = "mean_squared_error")

#Training
epochs_hist = my_model.fit(x_train, y_train, epochs=100)

print(epochs_hist.history.keys)
print(my_model.summary())
print(kilogrames)
print(pounds)

def predict(model:object, kg_input:float, kilogrames_array:object, pounds_array:object):
    """
    DOCSTRING:
        This method calculate if the transformed kg to lb given value is correct or not, if it is correct so the value will be saved as new data in the kg and lb datasets, if the result is incorrect then it will be discarted.
    
    Attributes:
        model:object,
        kg_input:float,
        kilogrames_array:object,
        pounds_array:object
     
    Return:
        result:str
    """
    
    model = model
    kg_input = kg_input
    kilogrames = kilogrames_array
    pounds = pounds_array

    predicted_pounds = model.predict([kg_input])
    real_pounds = kg_input * 2.20462
    
    if predicted_pounds == real_pounds:
        kg_added = np.array([kg_input])
        lb_added = np.array([predicted_pounds])
        kilogrames = np.concatenate(kilogrames, kg_added)
        pounds = np.concatenate(pounds, lb_added)
        succesfully:bool = True
    else:
        succesfully:bool = False
    
    result:str = f"Kg input:{kg_input}\nPredicted Lb:{predicted_pounds}\nReal Lb:{real_pounds}\nSuccesfully:{succesfully}"
    
    return result
    
print(predict(my_model, 96, kilogrames, pounds))
