import pandas as pd 
import csv
import re

data_csv = csv.writer(open('./data/train_new.csv', 'a', encoding='utf-8', newline=''))

with open('./data/Train.csv', 'r', encoding='latin1') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data = list(readCSV)

    for row in data:
        label = row[1]
        row = re.sub(r'[^\x00-\x7F]+', ' ', str(row[0]))
        row.strip()
        row = row.encode("ascii", errors="ignore").decode()
        if row is '':
            continue
        data_csv.writerow([row, label])

