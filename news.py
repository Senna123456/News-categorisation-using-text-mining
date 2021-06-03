

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123456)
dataset = pd.read_csv("uci-news-aggregator.csv")
dataset.head()
dataset['CATEGORY'].unique()
dataset['CATEGORY'].value_counts().plot(kind="bar")
plt.show()
import re
import string

def clean_text(s):
    s = s.lower()
    for ch in string.punctuation:                                                                                                     
        s = s.replace(ch, " ") 
    s = re.sub("[0-9]+", "||DIG||",s)
    s = re.sub(' +',' ', s)        
    return s

dataset['TEXT'] = [clean_text(s) for s in dataset['TITLE']]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import csv
# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(dataset['TEXT'])

# for Tfidf (we have tried and the results aren't better)
#tfidf = TfidfVectorizer()
#x = tfidf.fit_transform(dataset['TEXT'].values)

encoder = LabelEncoder()
y = encoder.fit_transform(dataset['CATEGORY'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# take a look at the shape of each of these
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


nb = MultinomialNB()
nb.fit(x_train, y_train)


results_nb_cv = cross_val_score(nb, x_train, y_train, cv=10)
print(results_nb_cv.mean())

nb.score(x_test, y_test)
x_test_pred = nb.predict(x_test)
confusion_matrix(y_test, x_test_pred)
print(classification_report(y_test, x_test_pred, target_names=encoder.classes_))



def predict_cat(title):
    cat_names = {'b' : 'business', 't' : 'science and technology', 'e' : 'entertainment', 'm' : 'health'}
    cod = nb.predict(vectorizer.transform([title]))
    return cat_names[encoder.inverse_transform(cod)[0]]


#value=input("Enter a head line")
#print(predict_cat(value))
fields = ['News', 'Category']
rows =[] 
with open('dataload.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            predict_cat(row[0])
            print(f'\t{row[0]} , {predict_cat(row[0])} .')
            rows.append([row[0],predict_cat(row[0])])
            with open('loaddata.csv', 'w') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                    
                # writing the fields 
                csvwriter.writerow(fields) 
                    
                # writing the data rows 
                csvwriter.writerows(rows)
            line_count += 1
    print(f'Processed {line_count} lines.')

print(rows)