
import Webscraping.ReadCsv as ReadCsv
import numpy as np


def sort_dict(dict_data):
    sorted_dict = dict(sorted(dict_data.items(), key=lambda item: item[1],reverse=True))
    return sorted_dict


def filter_dict_percentile(dict_data, percentile):
    values = list(dict_data.values())
    threshold = np.percentile(values, percentile)
    filtered_dict = {key: value for key, value in dict_data.items() if value >= threshold}
    return filtered_dict

