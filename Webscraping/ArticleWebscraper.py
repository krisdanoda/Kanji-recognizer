from selenium import webdriver
import numpy as np
import keyboard
import csv
import ReadCsv
driver = webdriver.Chrome()

data_labels = np.load('../kkanji-labels.npz')
labels = data_labels['arr_0']
labels = np.unique(labels)

d = dict()
try:
    d = ReadCsv.open_csv('kanji_freq.csv')
except FileNotFoundError:
    print("File not found")



def kanji(kaniUnicode):
    return chr(int(kaniUnicode[2:], 16))


def add_to_dict(dict_data):
    dta = []
    print(url)
    driver.get(base_url + url.__str__() + '.html')
    dta = np.append(dta, driver.find_elements_by_class_name("body-text"))
    dta = np.append(dta, driver.find_elements_by_class_name("content--summary"))
    dta = np.append(dta, driver.find_elements_by_class_name("content--summary-more"))

    if dta.__len__() != 0:
        print("not empty")
    for i in dta:
        for char in i.text:
            if char in u_kanjis:

                unicode_code_point = ord(char)
                unicode_hex = format(unicode_code_point, 'X')
                unicode_with_format = 'U+' + unicode_hex
                char = unicode_with_format

                if dict_data.get(char) != None:
                    dict_data[char] = dict_data[char] + 1
                else:
                    dict_data[char] = 1


u_kanjis = [kanji(i) for i in labels]

##urls = ['https://www3.nhk.or.jp/news/html/20230516/k10014068841000.html',
##        'https://www3.nhk.or.jp/news/html/20230516/k10014068841000.html',
##        'https://www3.nhk.or.jp/news/html/20230516/k10014068841000.html']

base_url = 'https://www3.nhk.or.jp/news/html/20230501/k'

a = 10014067751000
b = 10014060001000


for url in range(b, 20000000000090, 10000):
    if keyboard.is_pressed('q'):  # if key 'q' is pressed
        print('You Pressed A Key!')
        break
    else:
        add_to_dict(d)



driver.close()

ReadCsv.write_dict_to_csv("kanji_freq.csv", d)
