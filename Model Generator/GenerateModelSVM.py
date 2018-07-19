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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

print('Loading Models...\n')

training_Data = pd.read_csv('Train.csv')
testing_Data = pd.read_csv('Test.csv')

svm_pipeline_final = Pipeline([
        ('svm_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,2),use_idf=True,smooth_idf=False)),
        ('svm_clf',CalibratedClassifierCV(svm.LinearSVC(penalty='l2')))
	])

svm_pipeline_final.fit(training_Data['Statement'],training_Data['Label'])
predicted_svm_final = svm_pipeline_final.predict(testing_Data['Statement'])
np.mean(predicted_svm_final == testing_Data['Label'])

pickle.dump(svm_pipeline_final,open('Model.sav','wb'))

