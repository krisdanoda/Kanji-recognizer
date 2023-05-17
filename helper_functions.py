import cv2

def resize(image, new_dim):
    dim = (new_dim, new_dim)
    return cv2.resize(image, dim, 1)

def to_kanji(kanjiUnicode):
    return chr(int(kanjiUnicode[2:], 16))