import csv

# change Sunday-1, Monday-2, Tuesday-3....Saturday-7
def day_change(data):
    for x in range(len(data)):
        if data[x][1] == "Sunday":
            data[x][1] = 1
        elif data[x][1] == "Monday":
            data[x][1] = 2
        elif data[x][1] == "Tuesday":
            data[x][1] = 3
        elif data[x][1] == "Wednesday":
            data[x][1] = 4
        elif data[x][1] == "Thursday":
            data[x][1] = 5
        elif data[x][1] == "Friday":
            data[x][1] = 6
        elif data[x][1] == "Saturday":
            data[x][1] = 7
        else:
            print ("Day Datatype error...")
            data[x][1] = 8


# change work_flow number
def work_flow_change(data):
    for x in range(len(data)):
        data[x][3] = int(data[x][3][10:])


# file name update

def file_name_change(data):
    for x in range(len(data)):
        data[x][4] = int(data[x][4][5:])


# convert string to number for matrix
def int_convert(data):
    for x in range(len(data)):
        for y in range(len(data[0])):
            data[x][y] = float(data[x][y])
    return data


def data_converter(data):
    day_change(data)
    work_flow_change(data)
    file_name_change(data)
    data = int_convert(data)
    return data


def open_file(address):
    with open(address) as Csvfile:
        reader = csv.reader(Csvfile)
        orign_data = list(reader)
        if address == 'network_backup_dataset.csv':
            orign_data = list(orign_data[1:])
            orign_data = data_converter(orign_data)
        else:
            orign_data = int_convert(orign_data)
        return orign_data