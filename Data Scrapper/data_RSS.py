from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import csv
import re

path = './ST/'
files = [f for f in listdir(path) if isfile(join(path, f))]

data_csv = csv.writer(open('train_new.csv', 'a', encoding='utf-8', newline=''))
data_csv.writerow(['Statement', 'Label'])

print(files)

for f in files:
    file = open(path+f, 'r', encoding='latin1')
    data = file.read()
    soup = BeautifulSoup(data, 'lxml')

    items = soup.find('rss')

    if items is not None: 
        title_list = items.find_all('title')
        for title in title_list:
            title_edit = title.text
            title_edit = re.sub(r'[^\x00-\x7F]+', ' ', title_edit)
            title_edit.strip()
            title_edit = title_edit.encode("ascii", errors="ignore").decode()
            data_csv.writerow([title_edit, 'TRUE'])
