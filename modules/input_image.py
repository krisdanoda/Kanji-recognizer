#/usr/bin/python
import cv2
import matplotlib.pyplot
from tensorflow import keras
import numpy as np
import pathlib
from modules import helper_functions
import tensorflow as tf
from modules import webscraping
from modules import accuracy_history


#LOAD LABELS
def process_image(default_image = 0, path ="input_images", save_accuracy = False, kanji_unicode = None):
    Z = helper_functions.load_labels(-1)
    X = helper_functions.load_images(-1)
    #FIND IMAGE
    input_images = pathlib.Path(path)

    
    input_images_strings = [str(item) for item in input_images.iterdir()]
    

    img = cv2.imread(input_images_strings[default_image])   # Read the image


    #RESIZE
    smallImg = helper_functions.resize(img, 64)

    #SHAPE IMAGE
    new_image = cv2.cvtColor(smallImg, cv2.COLOR_RGB2GRAY)
    new_image = cv2.bitwise_not(new_image)
    new_image = np.array(new_image)
    threshold = 16
    new_image = np.where(new_image >= threshold, 255, 0) #Turn images to black and white
    new_image=new_image/255


    #PREDICTION
    model = keras.models.load_model('Journal/models/withdropout_and_larger_kernel_bw')

    dim_img = tf.expand_dims(new_image, 0)
    predictions = model.predict(dim_img)

    predicted_class_index = predictions.argmax(axis=-1)[0]

    Z = helper_functions.clean_data(Z, X)[0]
    labels = np.unique(Z)

    if kanji_unicode != None:
        predicted_class_index = np.where(labels == str(kanji_unicode))[0][0]


    predicted_class_probability = predictions[0][predicted_class_index]
    # SAVE ACCURACY
    if save_accuracy:
        accuracy_history.save_accuracy(str(kanji_unicode), predicted_class_probability * 100)

        print("Accuracy saved to history", predicted_class_probability * 100)

    #WEBSCRAPE MEANING

    return_list = [helper_functions.to_kanji(labels[predicted_class_index])]
    return_list.append(webscraping.get_meaning(helper_functions.to_kanji(labels[predicted_class_index])))
    return_list.append(round(predicted_class_probability*100,2))

    return return_list



