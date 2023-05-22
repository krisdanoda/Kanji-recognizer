import dataCleaning
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Webscraping.ReadCsv as read_csv
import Webscraping.reduce_data as reduce_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn
from tqdm import tqdm
from keras.callbacks import ReduceLROnPlateau
from os import listdir, system


def load_data():
    try:
        data_imgs = np.load('../data/kkanj-imgs.npz')
        data_labels = np.load('../data/kkanji-labels.npz')
    except:
        data_imgs = np.load('data/kkanj-imgs.npz')
        data_labels = np.load('data/kkanji-labels.npz')
    imgs = data_imgs['arr_0']
    labels = data_labels['arr_0']
    return labels, imgs,


def clean_data_using_webscrapped_data(labels, imgs, ):

    try:
        dict_data = read_csv.add_nonocurring_kanjis("../Webscraping/kanji_freq.csv");
    except:
        dict_data = read_csv.add_nonocurring_kanjis("Webscraping/kanji_freq.csv")

    print("amount of kanji before cleaning " + len(dict_data).__str__())

    # Remove 50 precent of the data, apparently 50% of characters of some characters
    dict_data = reduce_data.filter_dict_percentile(dict_data, 50)

    print("amount of kanji after cleaning " + len(dict_data).__str__())

    mask = [key in dict_data for key in labels]  # Create a mask based on the presence of keys in the dictionary

    print("amount of data before cleaning " + len(labels).__str__())
    imgs = [imgs[i] for i, include in enumerate(mask) if include]  # Apply the mask to array x
    labels = [labels[i] for i, include in enumerate(mask) if include]  # Apply the mask to array y

    print("amount of data after cleaning " + len(labels).__str__())

    return labels, imgs,


def data_normalization(imgs):
    x = np.array(imgs)
    x = x / 255
    return x


def one_hot_encode(Z):
    le = LabelEncoder()
    y = le.fit_transform(Z)
    y = to_categorical(y, len(np.unique(Z)))
    return y


def define_model(Z, padding = "Same", activation = "relu", kernelsizes=None, filters = 32, dropout = False):
    if kernelsizes is None:
        kernelsizes = [3, 3, 1, 1]
    model = Sequential()
    model.add(Conv2D(filters= filters*1, kernel_size=(kernelsizes[0] , kernelsizes[0]), padding= padding, activation=activation, input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Conv2D(filters= filters*2, kernel_size=(kernelsizes[1], kernelsizes[1]), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Conv2D(filters= filters*3, kernel_size=(kernelsizes[2], kernelsizes[2]), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Conv2D(filters= filters*3, kernel_size=(kernelsizes[3], kernelsizes[3]), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(Z)), activation="softmax"))
    return model


def fit_model(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    #model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # %%
    model.summary()
    # %%
    History = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                        verbose=1, steps_per_epoch=x_train.shape[0] // batch_size)
    return History


def run_model(name, epochs, batchsize, useContoursFiltering = True, useDataFiltering = True, padding = "Same", activation = "relu", kernelsizes=None, filters = 32):
    labels, imgs = load_data()

    filtered_labels, filtered_imgs = dataCleaning.remove_min_occurences(labels, imgs)
    if useContoursFiltering:
        filtered_labels, filtered_imgs = dataCleaning.remove_by_contours(filtered_labels, filtered_imgs)
    if useDataFiltering:
        filtered_labels, filtered_imgs = clean_data_using_webscrapped_data(filtered_labels, filtered_imgs)
    X = data_normalization(filtered_imgs)
    Y = one_hot_encode(filtered_labels)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)

    model = define_model(filtered_labels, padding = padding, activation = activation, kernelsizes = kernelsizes, filters= filters)

    history =fit_model(model, x_train, y_train, x_test, y_test, batchsize, epochs)

    model.save('models/' + name )

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])
    plt.show()
    return model


##run_model("control-with-kanji-filtering", 10, 200, useContoursFiltering=False, useDataFiltering= True)
