#/usr/bin/python
from matplotlib import pyplot as plt
import cv2
from tensorflow import keras
import numpy as np
import pathlib
import helper_functions
import tensorflow as tf
import webscraping

#LOAD LABELS
def give_image_meaning():
    Z = helper_functions.load_labels(14000)

    #FIND IMAGE
    input_images = pathlib.Path("input_images")
    list(input_images.iterdir())

    input_images_strings = [str(item) for item in input_images.iterdir()]
    

    img = cv2.imread(input_images_strings[0])   # Read the image


    #RESIZE
    smallImg = helper_functions.resize(img, 64)

    #SHAPE IMAGE
    new_image = cv2.cvtColor(smallImg, cv2.COLOR_RGB2GRAY)
    new_image = cv2.bitwise_not(new_image)
    new_image=np.array(new_image)
    new_image=new_image/255

    #PREDICTION
    model = keras.models.load_model('../Kanji-recognizer/saved_sequential_model')

    dim_img = tf.expand_dims(new_image, 0)
    predictions = model.predict(dim_img)
    predicted_classes = predictions.argmax(axis=-1)


    #WEBSCRAPE MEANING

    return_list = [helper_functions.to_kanji(np.unique(Z)[predicted_classes][0])]
    return_list.append(webscraping.get_meaning(helper_functions.to_kanji(np.unique(Z)[predicted_classes][0])))
    
    return return_list



