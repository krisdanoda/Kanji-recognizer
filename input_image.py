#/usr/bin/python
from matplotlib import pyplot as plt
import cv2
from tensorflow import keras
import numpy as np
import pathlib
import helper_functions
import tensorflow as tf
import webscraping
import accuracy_history

#LOAD LABELS
def give_image_meaning(default_image = 0, path = "input_images"):
    Z = helper_functions.load_labels(-1)

    #FIND IMAGE
    input_images = pathlib.Path(path)
    list(input_images.iterdir())

    input_images_strings = [str(item) for item in input_images.iterdir()]
    

    img = cv2.imread(input_images_strings[default_image])   # Read the image


    #RESIZE
    smallImg = helper_functions.resize(img, 64)

    #SHAPE IMAGE
    new_image = cv2.cvtColor(smallImg, cv2.COLOR_RGB2GRAY)
    new_image = cv2.bitwise_not(new_image)
    new_image=np.array(new_image)
    new_image=new_image/255

    print(new_image)
    print(new_image.shape)
    #PREDICTION
    model = keras.models.load_model('saved_sequential_model')

    dim_img = tf.expand_dims(new_image, 0)
    predictions = model.predict(dim_img)
    predicted_classes = predictions.argmax(axis=-1)

    predicted_class_index = predictions.argmax(axis=-1)[0]
    predicted_class_probability = predictions[0][predicted_class_index]

    #SAVE ACCURACY

    accuracy_history.save_accuracy(np.unique(Z)[predicted_class_index], predicted_class_probability * 100)


    #WEBSCRAPE MEANING

    return_list = [helper_functions.to_kanji(np.unique(Z)[predicted_classes][0])]
    return_list.append(webscraping.get_meaning(helper_functions.to_kanji(np.unique(Z)[predicted_classes][0])))
    return_list.append(predicted_class_probability*100)

    return return_list



