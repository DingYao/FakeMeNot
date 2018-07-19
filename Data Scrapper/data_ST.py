import requests
import time
import re
import csv
from bs4 import BeautifulSoup

req_main = 'https://www.straitstimes.com/'
links = ['news/world/rss.xml','news/business/rss.xml','news/sport/rss.xml','news/lifestyle/rss.xml','news/opinion/rss.xml','news/singapore/rss.xml','news/politics/rss.xml','news/asia/rss.xml','news/tech/rss.xml','news/forum/rss.xml','news/multimedia/rss.xml']
req_list = []

file = csv.writer(open('data_ST.csv', 'a'))
file.writerow(['Title', 'Link', 'Content'])

for link in links:
    req_list.append(req_main + link)


for req_link in req_list:
    page = requests.get(req_link)
    soup = BeautifulSoup(page.text, 'lxml')
    items_raw_list = soup.find('rss')

    if items_raw_list is not None:
        items_list = items_raw_list.find_all('item')
        title_list = []
        article_link = []

        for item in items_list:
            title_list.append(item.find('title').text)
            article_link.append(item.find('guid').text)

        i = 0
        for i in range(0,len(items_list)):
            print(title_list[i])
            print(article_link[i])

            content_page = requests.get(article_link[i])
            content_soup = BeautifulSoup(content_page.text, 'lxml')
            content_raw_list = content_soup.find('div', {'class':'group-ob-readmore'})

            #print(content_raw_list)
            if content_raw_list is None:
                continue

            content_list_p = content_raw_list.find_all('p')

            content_list = []
            for content in content_list_p:
                content_list.append(content.text)

            file.writerow([title_list[i], article_link[i], content_list])
            time.sleep(0.5)
