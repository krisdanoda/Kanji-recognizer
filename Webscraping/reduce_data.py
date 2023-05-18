import ReadCsv
import numpy as np

dict_data = ReadCsv.open_csv('kanji_freq.csv')

data_imgs = np.load('kkanj-imgs.npz')
data_labels = np.load('kkanji-labels.npz')

imgs = data_imgs['arr_0']
labels = data_labels['arr_0']


def sort_dict(dict_data):
    sorted_dict = dict(sorted(dict_data.items(), key=lambda item: item[1],reverse=True))
    return sorted_dict


def filter_dict_percentile(dict_data, percentile):
    values = list(dict_data.values())
    threshold = np.percentile(values, percentile)
    filtered_dict = {key: value for key, value in dict_data.items() if value >= threshold}
    return filtered_dict


filter_dict = filter_dict_percentile(dict_data, 30)


print(filter_dict_percentile(dict_data, 30))

