import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

print('Loading Models...\n')

training_Data = pd.read_csv('Train.csv')
testing_Data = pd.read_csv('Test.csv')

nb_final = Pipeline([
        ('nb_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),
        ('nb_clf',MultinomialNB())])
    
nb_final.fit(training_Data['Statement'],training_Data['Label'])
predicted_nb_final = nb_final.predict(testing_Data['Statement'])
np.mean(predicted_nb_final == testing_Data['Label'])

pickle.dump(nb_final,open('Model.sav','wb'))

