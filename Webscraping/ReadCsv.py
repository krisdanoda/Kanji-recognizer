import csv
import numpy as np


def write_dict_to_csv(file_name, dict_data):
    with open(file_name, "w", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dict_data.keys())

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerow(dict_data)
        print('Done writing dict to a csv file')


def open_csv(name):
    with open(name, "r") as infile:
        # Create a reader object
        reader = csv.DictReader(infile)

        # Iterate through the rows
        for row in reader:
            # Convert the values to integers
            converted_row = {key: int(value) for key, value in row.items()}

    return converted_row


def add_nonocurring_kanjis(name, save = False):
    data_imgs = np.load('data/kkanj-imgs.npz')
    data_labels = np.load('data/kkanji-labels.npz')

    # Access the arrays stored in the .npz files
    imgs = data_imgs['arr_0']
    labels = data_labels['arr_0']

    dict_data = open_csv(name)

    for k in labels:
        if dict_data.get(k) is None:
            dict_data[k] = 0


    if save:
        write_dict_to_csv(name, dict_data)

    return dict_data
