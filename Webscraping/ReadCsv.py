import csv

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