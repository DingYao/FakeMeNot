from telegram.ext import Updater, MessageHandler, CommandHandler, Filters
from googleapiclient.discovery import build
import pprint
import requests
from bs4 import BeautifulSoup
from rake_nltk import Rake
import re
import pickle

my_api_key = "AIzaSyB2U2Ue27sLYd-ZbRSU2ToFRoC6Oq6jwac"
my_cse_id = "015858408429755719149:zqe0qao-lcm"

blacklist = ['newnation.sg', '70news.wordpress.com','Abcnews.com.co', 'infowars.com','yournewswire.com','rilenews.com', 'statestimesreview.com', 'allsingaporestuff.com', 'forums.hardwarezone.com']
whitelist = ['straitstimes.com','channelnewsasia.com','gov.sg/factually','reuters.com','snopes.com']

def is_fake(str):
    print('Predicting \'' + str + '\'')
    model = pickle.load(open('Model.sav', 'rb'))
    predict = model.predict([str])
    proba = model.predict_proba([str])
    if predict[0] == True:
        res = '*' + repr(predict[0]) + '* with *%.2f' % round((proba[0][1]*100),2) + '%* ' + 'Confidence.'
    else:
        res = '*' + repr(predict[0]) + '* with *%.2f' % round((100 - (proba[0][1]*100)),2) + '%* ' + 'Confidence.'
    print('Statement is ' + repr(predict[0]) + ', ' + 'Probability is ' + repr(proba[0][1]) + '.\n')
    return res

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    # print(res)
    return res['items']

def main_search(input):
    results = google_search(input, my_api_key, my_cse_id, num=10)
    str = ''
    i = 0
    for result in results:
        # print('==============================')
        if i > 2:
            break
        str = str + result.get('title') + '\n' + result.get('link') + '\n\n'
        i = i + 1
        # print(result.get('title'))
        # print(result.get('link'))
    return str

def get_important_words(str):
    r = Rake()
    r.extract_keywords_from_text(str)
    return r.get_ranked_phrases()

def from_twitter_url(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'lxml')
    content = [p.text for p in soup.find_all('p', class_='tweet-text')]
    string = content[0]
    # print(string)
    string = string.replace(".", "")
    string = string.replace(",", "")
    string = string.replace("#", "")
    string = re.sub(r"http\S+", "", string, flags=re.MULTILINE)
    string = re.sub(r"pictwittercom/\S+", "", string, flags=re.MULTILINE)
    # print(string)
    return string

def is_safe(input):
    # print(input)
    if any(s in input for s in blacklist):
        return False
    return True

def start(bot, update):
    print('Received Start Request.\n')
    update.message.reply_text('Welcome to *Fake Me Not*, developed by Glenice, Ding Yao and Kok Yin for codextreme 2018!\n\nPlease enter a statement or the URL of a Twitter post you wish to analyse.', parse_mode='Markdown')

def get_input(bot, update):
    print('Received Input: ' + update.message.text )
    input = update.message.text
    # update.message.reply_text('Please wait while we process the data. If there is no response within 2 minutes, please retry.')

    reply = ''
    # print(is_safe(input))
    if is_safe(input) is True:
        # print('safe is true')
        if 'twitter.com' in input:
            update.message.reply_text(text='_Our Prediction:_\n\nPost is ' + is_fake(from_twitter_url(input)), parse_mode='Markdown')
            impt_words = get_important_words(from_twitter_url(input))
            if len(impt_words) >= 2:
                reply = '_Recent News:_\nDetected Keywords: [ ' + impt_words[0] + ' ' + impt_words[1] + ' ]\n\n'
            elif len(impt_words) < 2:
                reply = '_Recent News:_\n\n'
            update.message.reply_text(text=reply + main_search(from_twitter_url(input)), parse_mode='Markdown', disable_web_page_preview=True)
        else:
            # print('not in twitter')
            update.message.reply_text(text='_Our Prediction:_\n\nStatement is ' + is_fake(input), parse_mode='Markdown')
            # print(impt_words)
            impt_words = get_important_words(input)
            if len(impt_words) >= 2:
                reply = '_Recent News:_\n*Detected Keywords:* [ ' + impt_words[0] + ' ' + impt_words[1] + ' ]\n\n'
            elif len(impt_words) < 2:
                reply = '_Recent News:_\n\n'
            update.message.reply_text(text=reply + main_search(input), parse_mode='Markdown', disable_web_page_preview=True)
    else:
        # print('safe is false')
        reply = '*Fake news site detected!* Please be wary in trusting its contents.'
        update.message.reply_text(text=reply, parse_mode='Markdown')

def main():
    updater = Updater(token='632890058:AAES4NVkmm6305aBsehQWJjy4MOZsz4uhI4')
    dispatcher = updater.dispatcher
    print('Bot Initialized.\n')

    # Add CommandHandler to Dispatcher
    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    # Add MessageHandler to Dispatcher
    input_handler = MessageHandler(Filters.text, get_input)
    dispatcher.add_handler(input_handler)

    # Start Bot
    updater.start_polling()

    # Run Bot until Terminated
    updater.idle()

if __name__ == '__main__':
  main()

