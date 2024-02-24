import numpy as np
import keras
from keras.models import save_model, load_model

"""Datasets"""
#Data arrays
pounds = []
kilogrames = np.array([1, 2, 5, 10, 17, 20, 50, 85, 100, 104, 177, 820, 1000, 1289, 5270])

for kg in kilogrames:
    pound = kg * 2.20462
    pounds.append(pound)
    
pounds = np.array(pounds)

#Loading dataset
x_train = kilogrames
y_train = pounds

"""Keras Model"""
try:
    model = loaded_model = load_model("C:/Users/raven/OneDrive/Escritorio/Coding/Python/Practices/AI_Practice/Trained_Models/kg_lb_model.h5")
except:
    #Architecture
    model = keras.models.Sequential([
        keras.layers.Dense(units=1, input_shape = [1], activation="relu"),
        keras.layers.Dense(1, activation="linear")
    ])

#Compiling
model.compile(optimizer = "adam", loss = "mean_squared_error")

#Training
epochs_hist = model.fit(x_train, y_train, epochs=100)

#Saving model
model.save("C:/Users/raven/OneDrive/Escritorio/Coding/Python/Practices/AI_Practice/Trained_Models/kg_lb_model.h5")

print(epochs_hist.history.keys)
print(model.summary())
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
        x_train = kilogrames
        y_train = pounds
        model.fit(x_train, y_train, epochs=10)
        model.save("C:/Users/raven/OneDrive/Escritorio/Coding/Python/Practices/AI_Practice/Trained_Models/kg_lb_model.h5") 
    else:
        succesfully:bool = False
    
    result:str = f"Kg input:{kg_input}\nPredicted Lb:{predicted_pounds}\nReal Lb:{real_pounds}\nSuccesfully:{succesfully}"
    
    return result
    
print(predict(model, 96, kilogrames, pounds))
print(predict(model, 105, kilogrames, pounds))
print(predict(model, 90, kilogrames, pounds))