import requests
import time
import re
import csv
from bs4 import BeautifulSoup
# Collect and parse first page
req_main = 'https://www.gov.sg'
req_all = 'https://www.gov.sg/factually/all/page-'
req_list = ['https://www.gov.sg/factually/all/page-1','https://www.gov.sg/factually/all/page-2','https://www.gov.sg/factually/all/page-3']

i = 4
for i in range(4, 51):
    req_list.append(req_all + str(i))

file = csv.writer(open('data.csv', 'a'))
file.writerow(['Title', 'Link', 'Content'])

for req_link in req_list:
    page = requests.get(req_link)
    soup = BeautifulSoup(page.text, 'lxml')

    discard_links = soup.find(class_='article-read rs_skip')
    if discard_links is not None:
        discard_links.decompose()

    # Pull all text from the BodyText div
    news_list = soup.find('div', {'class':'news-items listing list'})
    if news_list is None:
        continue

    article_title_list = news_list.find_all('h2')
    article_link_list = news_list.find_all('a')
    all_content = []

    for title in article_title_list:
        #links.append(link.attrs['href'])
        a = title.find('a')
        link = req_main + a.attrs['href']

        content_page = requests.get(req_main + a.attrs['href'])
        soup_content = BeautifulSoup(content_page.text, 'lxml')
        discard = soup_content.find(class_='control-wrap')
        if discard is not None:
            discard.decompose()

        discard = soup_content.find(class_='disclaimer')
        if discard is not None:
            discard.decompose()

        content_raw_list = soup_content.find('div', {'id':'readable'})
        if content_raw_list is None:
            continue

        content_list_p = content_raw_list.find_all('p')
        content_list = []

        for content in content_list_p:
            content_list.append(content.text)

        content_list = content_list[:-1]
        
        file.writerow([title.text, link, content_list])
        print(title.text)
        time.sleep(0.5)

#links_rm = list(set(links))
