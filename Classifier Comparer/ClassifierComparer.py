import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

print('########################################')
print('Team Mates Fake News Classifier Comparer')
print('   Developed during codextreme 2018')
print('########################################\n')


print('Updating NLTK Treebank...')
nltk.download('treebank')
print('NLTK Treebank Updated!\n')

print('Loading Data...')

training_Data = pd.read_csv('Train.csv')
print('Training Data Loaded.')
testing_Data = pd.read_csv('Test.csv')
print('Testing Data Loaded.\n')

countV = CountVectorizer()
tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)

print('Building Classifiers with Bag of Words Technique...')

nb_pipeline = Pipeline([
        ('NBCV',countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(training_Data['Statement'],training_Data['Label'])
predicted_nb = nb_pipeline.predict(testing_Data['Statement'])
np.mean(predicted_nb == testing_Data['Label'])

logR_pipeline = Pipeline([
        ('LogRCV',countV),
        ('LogR_clf',LogisticRegression())
        ])

logR_pipeline.fit(training_Data['Statement'],training_Data['Label'])
predicted_LogR = logR_pipeline.predict(testing_Data['Statement'])
np.mean(predicted_LogR == testing_Data['Label'])

svm_pipeline = Pipeline([
        ('svmCV',countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(training_Data['Statement'],training_Data['Label'])
predicted_svm = svm_pipeline.predict(testing_Data['Statement'])
np.mean(predicted_svm == testing_Data['Label'])


sgd_pipeline = Pipeline([
        ('svm2CV',countV),
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
        ])

sgd_pipeline.fit(training_Data['Statement'],training_Data['Label'])
predicted_sgd = sgd_pipeline.predict(testing_Data['Statement'])
np.mean(predicted_sgd == testing_Data['Label'])

random_forest = Pipeline([
        ('rfCV',countV),
        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        ])
    
random_forest.fit(training_Data['Statement'],training_Data['Label'])
predicted_rf = random_forest.predict(testing_Data['Statement'])
np.mean(predicted_rf == testing_Data['Label'])

def build_confusion_matrix(classifier):
    
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(training_Data['Statement'],training_Data['Label']):
        train_text = training_Data.iloc[train_ind]['Statement'] 
        train_y = training_Data.iloc[train_ind]['Label']
    
        test_text = training_Data.iloc[test_ind]['Statement']
        test_y = training_Data.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    
    return (print('Total Statements Classified:', len(training_Data)),
    print('Score:', sum(scores)/len(scores)),
    print('Score Length', len(scores)),
    print('Confusion Matrix:'),
    print(confusion))
    
print('Printing Confusion Matrices...')

print('Naive Bayes:')
build_confusion_matrix(nb_pipeline)

print('\nLogistic Regression:')
build_confusion_matrix(logR_pipeline)

print('\nLinear SVM:')
build_confusion_matrix(svm_pipeline)

print('\nStochastic Gradient Descent on Hinge Loss:')
build_confusion_matrix(sgd_pipeline)

print('\nRandom Forest:')
build_confusion_matrix(random_forest)

print('\nBuilding Classifiers with N-grams (5 Words)...')

nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(training_Data['Statement'],training_Data['Label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(testing_Data['Statement'])
np.mean(predicted_nb_ngram == testing_Data['Label'])

logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf',tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logR_pipeline_ngram.fit(training_Data['Statement'],training_Data['Label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(testing_Data['Statement'])
np.mean(predicted_LogR_ngram == testing_Data['Label'])

svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(training_Data['Statement'],training_Data['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(testing_Data['Statement'])
np.mean(predicted_svm_ngram == testing_Data['Label'])

sgd_pipeline_ngram = Pipeline([
         ('sgd_tfidf',tfidf_ngram),
         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
         ])

sgd_pipeline_ngram.fit(training_Data['Statement'],training_Data['Label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(testing_Data['Statement'])
np.mean(predicted_sgd_ngram == testing_Data['Label'])

random_forest_ngram = Pipeline([
        ('rf_tfidf',tfidf_ngram),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        ])
    
random_forest_ngram.fit(training_Data['Statement'],training_Data['Label'])
predicted_rf_ngram = random_forest_ngram.predict(testing_Data['Statement'])
np.mean(predicted_rf_ngram == testing_Data['Label'])

print('Printing Confusion Matrices...')

print('Naive Bayes:')
build_confusion_matrix(nb_pipeline_ngram)

print('\nLogistic Regression:')
build_confusion_matrix(logR_pipeline_ngram)

print('\nLinear SVM:')
build_confusion_matrix(svm_pipeline_ngram)

print('\nStochastic Gradient Descent on Hinge Loss:')
build_confusion_matrix(sgd_pipeline_ngram)

print('\nRandom Forest:')
build_confusion_matrix(random_forest_ngram)

print('\nPrinting Classification Reports...')

print('Naive Bayes:')
print(classification_report(testing_Data['Label'], predicted_nb_ngram))

print('\nLogistic Regression:')
print(classification_report(testing_Data['Label'], predicted_LogR_ngram))

print('\nLinear SVM:')
print(classification_report(testing_Data['Label'], predicted_svm_ngram))

print('\nStochastic Gradient Descent on Hinge Loss:')
print(classification_report(testing_Data['Label'], predicted_sgd_ngram))

print('\nRandom Forest:')
print(classification_report(testing_Data['Label'], predicted_rf_ngram))

print('\nFinding Best Parameters for N-gram Classifiers (5 Words)...')
print('Printing Results...')

parameters = {'nb_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'nb_tfidf__use_idf': (True, False),
               'nb_tfidf__smooth_idf': (True, False)
}

gs_clf = GridSearchCV(nb_pipeline_ngram, parameters, n_jobs=-1, return_train_score=True)
gs_clf = gs_clf.fit(training_Data['Statement'][:10000],training_Data['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_
print('Naive Bayes:')
print (pd.DataFrame(gs_clf.cv_results_)[['params', 'mean_train_score', 'mean_test_score']])
print ('Best Parameters: ', gs_clf.best_params_)
print ('Best Score: ', gs_clf.best_score_)

parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'LogR_tfidf__use_idf': (True, False),
               'LogR_tfidf__smooth_idf': (True, False)
}

gs_clf = GridSearchCV(logR_pipeline_ngram, parameters, n_jobs=-1, return_train_score=True)
gs_clf = gs_clf.fit(training_Data['Statement'][:10000],training_Data['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_
print('\nLogistic Regression:')
print (pd.DataFrame(gs_clf.cv_results_)[['params', 'mean_train_score', 'mean_test_score']])
print ('Best Parameters: ', gs_clf.best_params_)
print ('Best Score: ', gs_clf.best_score_)

parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'svm_tfidf__use_idf': (True, False),
               'svm_tfidf__smooth_idf': (True, False),
               'svm_clf__penalty': ('l2', 'l2'),
}

gs_clf = GridSearchCV(svm_pipeline_ngram, parameters, n_jobs=-1, return_train_score=True)
gs_clf = gs_clf.fit(training_Data['Statement'][:10000],training_Data['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_
print('\nLinear SVM:')
print (pd.DataFrame(gs_clf.cv_results_)[['params', 'mean_train_score', 'mean_test_score']])
print ('Best Parameters: ', gs_clf.best_params_)
print ('Best Score: ', gs_clf.best_score_)

parameters = {'sgd_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'sgd_tfidf__use_idf': (True, False),
}

gs_clf = GridSearchCV(sgd_pipeline_ngram, parameters, n_jobs=-1, return_train_score=True)
gs_clf = gs_clf.fit(training_Data['Statement'][:10000],training_Data['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_
print('\nStochastic Gradient Descent on Hinge Loss:')
print (pd.DataFrame(gs_clf.cv_results_)[['params', 'mean_train_score', 'mean_test_score']])
print ('Best Parameters: ', gs_clf.best_params_)
print ('Best Score: ', gs_clf.best_score_)

parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'rf_tfidf__use_idf': (True, False),
               'rf_clf__max_depth': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
}

gs_clf = GridSearchCV(random_forest_ngram, parameters, n_jobs=-1, return_train_score=True)
gs_clf = gs_clf.fit(training_Data['Statement'][:10000],training_Data['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_
print('\nRandom Forest:')
print (pd.DataFrame(gs_clf.cv_results_)[['params', 'mean_train_score', 'mean_test_score']])
print ('Best Parameters: ', gs_clf.best_params_)
print ('Best Score: ', gs_clf.best_score_)

