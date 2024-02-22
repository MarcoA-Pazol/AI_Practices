import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import TensorBoard
from keras.models import load_model, save_model
import matplotlib.pyplot as plt
import numpy as np

#Image generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

#Image Datasets
train_generator = train_datagen.flow_from_directory(
    "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/food_dataset/train",
    batch_size=5,
    target_size=(150, 150),
    class_mode="binary",
    color_mode="grayscale",
    shuffle=False,    
)

validation_generator = validation_datagen.flow_from_directory(
    "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/food_dataset/validation",
    batch_size=5,
    target_size=(150, 150),
    class_mode="binary",
    color_mode="grayscale",
    shuffle=False    
)

#Steps per epoch and validation steps
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.samples


"""MODELS: 3 Diferent models for testing their parameters and values to find the most efficient model."""
model_one = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(4, (3, 3), activation="relu", input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_two = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_three = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")    
])

#Compiling
model_one.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True)
model_two.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True)
model_three.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True)

model_one.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=[TensorBoard("C:/Users/raven/OneDrive/Escritorio/Coding/Python/Practices/AI_Practices/Logs/model_one_binary_classification")]
)

model_two.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=[TensorBoard("C:Users/ravne/OneDrive/Escritorio/Coding/Python/Practices/AI_Practices/Logs/model_two_binary_classification")]
)

model_three.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=[TensorBoard("C:/Users/raven/OneDrive/Escritorio/Coding/Python/Practices/AI_Practices/Logs/model_three_binary_classification")]
)

def predict(model:object, path:str):
    """
    DOCSTRING:
        This method predict if the image is a healthy or unhealthy food using 3 diferent models that we give as first parameter when we call this method and it returns the accuracy, prediction result and image path.
    
    Attributes:
        model:object
        path:str
        
    Return:
        result:str
    """
    try:
        img_path = path
        model = model
        img = image.load_img(img_path, target_size=(150, 150), color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        prediction = model.predict(img_array)
    except Exception as e:
        print(f"Exception: {e}")
    
    threshold = 0.5
    if prediction[0] < threshold:
        predicted_class = "Healthy Food"
    elif prediction[0] > threshold:
        predicted_class = "Unhealthy Food"
    else:
        predicted_class = "Draw"
 
    result = f"Result: {predicted_class}\nOutput value: {prediction}\nPath: {img_path}"
    return result

print("\nMODEL ONE:")
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/apple.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/apple2.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/grappes.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/guava.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/hamburguer1.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/kiwi.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/orange1.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/orange2.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/pizza1.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/pumpkin.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/rice1.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/salad1.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/tacos1.jpg"))
print(predict(model_one, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/tuna1.jpg"))

print("\nMODEL TWO:")
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/apple.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/apple2.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/grappes.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/guava.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/hamburguer1.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/kiwi.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/orange1.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/orange2.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/pizza1.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/pumpkin.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/rice1.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/salad1.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/tacos1.jpg"))
print(predict(model_two, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/tuna1.jpg"))

print("\nMODEL THREE:")
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/apple.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/apple2.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/grappes.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/guava.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/hamburguer1.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/kiwi.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/orange1.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/orange2.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/pizza1.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/pumpkin.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/rice1.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/salad1.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/tacos1.jpg"))
print(predict(model_three, "C:/Users/raven/OneDrive/Escritorio/Coding/Python/Fundaments/AI_Fundaments/Tensorflow/DataSets/prediction_images/tuna1.jpg"))