import requests
from bs4 import BeautifulSoup
import csv

statestimes_all = 'http://statestimesreview.com/page/'
statestimes_list = ['http://statestimesreview.com/page/1/']
i = 2
for i in range(2, 353):
    statestimes_list.append(statestimes_all + str(i) + '/')
    # print(statestimes_list[i-1])
file = csv.writer(open('train_ST.csv', 'a', newline=''))
file.writerow(['Statement', 'Label'])
for statestimes_link in statestimes_list:
    page = requests.get(statestimes_link)
    soup = BeautifulSoup(page.text, 'html.parser')
    last_links = soup.find_all(class_='post-thumbnail')
    for lastlink in last_links:
        lastlink.decompose()
    last_links2 = soup.find_all(class_='penci-cat-name')
    for lastlink in last_links2:
        lastlink.decompose()
    title_list = soup.find(class_='masonry penci-masonry')
    title_list_items = title_list.find_all('a')
    for title in title_list_items:
        names = title.string
        if names is None:
            continue
        print(names)
        try:
            file.writerow([names, 'False'])
        except:
            pass