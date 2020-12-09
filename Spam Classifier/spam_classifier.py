# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:14:01 2020

@author: Lucky_Rathod
"""

#### SPAM Classifier -- These spam classifier will be able to predict whether the message is spam or not

################################ SPAM CLASSIFIER USING BOW MODEL #############################

import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])

### Data cleaning and Text Preprocessing

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
wordnet = WordNetLemmatizer()
corpus=[]

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
### Creating Bag of Words Model

from sklearn.feature_extraction.text import CountVectorizer

### High Frequency words will be extracted from the corpus . And it will be in columns of matrix given below
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

### Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

### Training Model with Naive Bayes Classifier 

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()
spam_detect_model = spam_detect_model.fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)


################################ SPAM CLASSIFIER USING TF-IDF MODEL #############################

#### TFIDF is not performing well than BOW 

#### Accuracy and recall is not good while using TFIDF 

import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])

### Data cleaning and Text Preprocessing

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
wordnet = WordNetLemmatizer()
corpus=[]

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
### Creating TFIDF Model

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray() 

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

### Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

### Training Model with Naive Bayes Classifier 

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()
spam_detect_model = spam_detect_model.fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)