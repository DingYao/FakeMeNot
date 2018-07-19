import requests
from bs4 import BeautifulSoup
import csv

newnation_all = 'http://newnation.sg/'
newnation_list = ['http://newnation.sg/2011/']
newnation_list2 = ['http://newnation.sg/2010/12/']
i = 2012
for i in range(2012, 2018):
    newnation_list.append(newnation_all + str(i) + '/')
j = 1
k = 0
for j in range(1, 12):
    for k in range(0, 7):
        newnation_list2.append(newnation_list[k] + str(j) + '/') 
file = csv.writer(open('train.csv', 'a', newline=''))
file.writerow(['Statement', 'Label'])
for newnation_link in newnation_list2:
    page = requests.get(newnation_link)
    soup = BeautifulSoup(page.text, 'html.parser')
    last_links = soup.find(class_='singletags')
    last_links2 = soup.find(class_='posted_in')
    last_links.decompose()
    last_links2.decompose()
    last_links3 = soup.find_all("h2")
    for lastlink4 in last_links3:
        lastlink4.decompose()
    title_list = soup.find(class_='col1')
    title_list_items = title_list.find_all('a')
    for title in title_list_items:
        names = title.get('title')
        if names is None:
            continue
        #print(names)
        try:
            file.writerow([names, 'False'])
        except:
            pass