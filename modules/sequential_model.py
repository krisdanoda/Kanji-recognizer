import numpy as np
import matplotlib.pyplot as plt
import Journal.Webscraping.ReadCsv as read_csv
from modules import data_cleaning
from modules import reduce_data as reduce_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Activation, Conv2D, MaxPooling2D
import tensorflow as tf
import random as rn


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
        dict_data = read_csv.add_nonocurring_kanjis("Journal/Webscraping/kanji_freq.csv");
    except:
        dict_data = read_csv.add_nonocurring_kanjis("Webscraping/kanji_freq.csv")



    # Remove 50 precent of the data, apparently 50% of characters of some characters
    dict_data = reduce_data.filter_dict_percentile(dict_data, 50)



    mask = [key in dict_data for key in labels]  # Create a mask based on the presence of keys in the dictionary


    imgs = [imgs[i] for i, include in enumerate(mask) if include]  # Apply the mask to array x
    labels = [labels[i] for i, include in enumerate(mask) if include]  # Apply the mask to array y

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


def define_model(Z, padding="Same", activation="relu", kernelsizes=None, filters=32, dropout=False):
    if kernelsizes is None:
        kernelsizes = [3, 3, 1, 1]
    model = Sequential()
    model.add(Conv2D(filters=filters * 1, kernel_size=(kernelsizes[0], kernelsizes[0]), padding=padding,
                     activation=activation, input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if dropout:
        model.add(Dropout(0.25))

    model.add(Conv2D(filters=filters * 2, kernel_size=(kernelsizes[1], kernelsizes[1]), padding='Same',
                     activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if dropout:
        model.add(Dropout(0.25))

    model.add(Conv2D(filters=filters * 3, kernel_size=(kernelsizes[2], kernelsizes[2]), padding='Same',
                     activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if dropout:
        model.add(Dropout(0.25))

    model.add(Conv2D(filters=filters * 3, kernel_size=(kernelsizes[3], kernelsizes[3]), padding='Same',
                     activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(Z)), activation="softmax"))
    return model


def gy_to_bw(gray_imgs):
    # Turns images from greyscale to black and white
    print("turning images to bw")
    gray_array = np.array(gray_imgs)
    threshold = 16
    return np.where(gray_array >= threshold, 255, 0)


def fit_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, learning_rate):
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    # %%
    model.summary()
    # %%
    History = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                        verbose=1, steps_per_epoch=x_train.shape[0] // batch_size)
    return History


def run_model(name, epochs, batchsize, use_contours_filtering=True, use_data_filtering=True, padding="Same",
              activation="relu", kernel_sizes=None, dropout=False, filters=32, learning_rate=0.00005, colors = "greyscale"):
    labels, imgs = load_data()

    filtered_labels, filtered_imgs = data_cleaning.remove_min_occurences(labels, imgs)
    if use_contours_filtering:
        filtered_labels, filtered_imgs = data_cleaning.remove_by_contours(filtered_labels, filtered_imgs)
    if use_data_filtering:
        filtered_labels, filtered_imgs = clean_data_using_webscrapped_data(filtered_labels, filtered_imgs)
    if colors == "bw" or colors == "BW":
        filtered_imgs = gy_to_bw(filtered_imgs)
    X = data_normalization(filtered_imgs)
    Y = one_hot_encode(filtered_labels)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)

    model = define_model(filtered_labels, padding=padding, activation=activation, kernelsizes=kernel_sizes,
                         filters=filters, dropout=dropout)

    history = fit_model(model, x_train, y_train, x_test, y_test, batchsize, epochs, learning_rate)

    model.save('models/' + name)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])
    plt.show()
    return model

##run_model("control-with-kanji-filtering", 10, 200, useContoursFiltering=False, useDataFiltering= True)
