import cv2
import numpy as np

import data_cleaning
import input_images
import random as rd

import sequential_model


def resize(image, new_dim):
    dim = (new_dim, new_dim)
    return cv2.resize(image, dim, 1)


def to_kanji(kanji_unicode):
    return chr(int(kanji_unicode[2:], 16))


def to_unicode(character):
    unicode_code_point = ord(character)
    unicode_hex = format(unicode_code_point, 'X')
    return 'U+' + unicode_hex

def load_labels(data_slice):
    data_labels = np.load('data/kkanji-labels.npz')
    labels = data_labels['arr_0']
    return labels[:data_slice]

def load_images(data_slice):
    data_imgs = np.load('data/kkanj-imgs.npz')
    imgs = data_imgs['arr_0']
    return imgs[:data_slice]

def random_kanji():
    Z = load_labels(-1)
    Z = clean_data(Z, load_images(-1))[0]
    U = np.unique(Z)
    label = rd.choice(U)
    kanji = to_kanji(label)
    return label, kanji

def clean_data(labels, images):

    Z, X = data_cleaning.remove_min_occurences(labels, images)
    Z, X = sequential_model.clean_data_using_webscrapped_data(Z, X)
    Z, X = data_cleaning.remove_by_contours(Z, X)
    return Z, X