import numpy as np
import keras
from keras.models import save_model, load_model
import matplotlib.pyplot as plt

"""Datasets"""
#Data arrays
kilogrames = np.array([1, 2, 5, 10, 17, 20, 50, 85, 100, 104, 177, 820, 1000, 1289, 5270, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 60, 70, 4, 3, 8, 12, 88, 94, 36, 55, 28, 71, 38])
pounds = np.array([])
for kg in kilogrames:
    lb = kg * 2.20462
    lb_array = np.array([lb])
    pounds = np.concatenate((pounds, lb_array))

#Loading dataset
x_train = kilogrames
y_train = pounds

"""Keras Model"""
try:
    model = loaded_model = load_model("C:/Users/raven/OneDrive/Escritorio/Coding/Python/Practices/AI_Practice/Trained_Models/kg_lb_model.h5")
except:
    #Architecture
    model = keras.models.Sequential([
        keras.layers.Dense(units=1, input_shape = [1])
    ])
# #Compiling
# model.compile(optimizer = keras.optimizers.Adam(0.5), loss = "mean_squared_error")

# #Training
# epochs_hist = model.fit(x_train, y_train, epochs=300)

# #Saving model
# model.save("C:/Users/raven/OneDrive/Escritorio/Coding/Python/Practices/AI_Practice/Trained_Models/kg_lb_model.h5")

# print(epochs_hist.history.keys)
# print(model.summary())
# print(kilogrames)
# print(pounds)

# """Training graphic for model accuracy progress during training"""
# plt.plot(epochs_hist.history["loss"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Lossing progress during model training")
# plt.show()

def predict(model:object, kg_input:float):
    """
    DOCSTRING:
        This method calculate if the transformed kg to lb given value is correct or not.
    
    Attributes:
        model:object,
        kg_input:float,
     
    Return:
        result:str
    """
    
    model = model
    kg_input = kg_input

    predicted_pounds = model.predict(np.array([kg_input]))
    real_pounds = kg_input * 2.20462
    
    threshold = 0.05
    if abs(predicted_pounds - real_pounds) <= threshold:
        succesfully:bool = True
        print(abs(predicted_pounds - real_pounds), threshold)
    else:
        succesfully:bool = False
        print(abs(predicted_pounds - real_pounds), threshold)
    
    result:str = f"Kg input:{kg_input}\nPredicted Lb:{predicted_pounds}\nReal Lb:{real_pounds}\nSuccesfully:{succesfully}"
    return result
    
print(predict(model, 96))
print(predict(model, 105))
print(predict(model, 90))
print(predict(model, 260))
print(predict(model, 400))
print(predict(model, 25))
print(predict(model, 10))
print(predict(model, 28))
print(predict(model, 30))
print(predict(model, 5200))