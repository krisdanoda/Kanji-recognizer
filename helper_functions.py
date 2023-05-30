import cv2
import numpy as np
import input_images
import random as rd


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
    U = np.unique(Z)
    label = rd.choice(U)
    kanji = to_kanji(label)
    return label, kanji