import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit

start = timeit.default_timer()

dataset = pd.read_csv("labeledTrainData.tsv",delimiter="\t",quoting = 3)

#cleaning datasets
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
cachedStopWords = set(stopwords.words("english"))
cachedStopWords.add('br')

for i in range(0,len(dataset)) :
    review = re.sub('[^a-zA-z]',' ',dataset['review'][i])
    review = review.replace('"','')
    review = review.replace('\\','')
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in cachedStopWords]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

tfidf = TfidfTransformer()
X = tfidf.fit_transform(X) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    

stop = timeit.default_timer()
print(f"\nRuntime : {stop-start}\n")

accuracy = (cm[0][0]+cm[1][1])/len(y_test)
print(f"Accuracy : {accuracy*100}%")