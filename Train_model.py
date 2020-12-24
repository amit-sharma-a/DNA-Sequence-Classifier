
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import os, pickle



def getData():
    human_data = pd.read_table('/home/amit/Documents/human_data.txt')
    chimp_data = pd.read_table('/home/amit/Documents/chimp_data.txt')
    dog_data = pd.read_table('/home/amit/Documents/dog_data.txt')
    return human_data, chimp_data, dog_data

def filter_Data(X):
    alfa = ["A","T","G","C"]
    new_enco = []
    for i in alfa:
        for j in alfa:
            new_enco.append(i+j)
    #print(new_enco)
    enco = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
    arr1 = []
    
    human = list(X[0]['sequence'])
    class_human = list(X[0]['class'])
    chimp = list(X[1]['sequence'])
    class_chimp = list(X[1]['class'])
    dog = list(X[2]['sequence'])
    class_dog = list(X[2]['class'])

    ####################human_filter
    for i in human:
        for j in i:
            if j not in alfa:
                class_human[human.index(i)] = np.nan
                human[human.index(i)] = np.nan

                break


    ###################chimp_filter
    for i in chimp:
        for j in i:
            if j not in alfa:
                class_chimp[chimp.index(i)] = np.nan
                chimp[chimp.index(i)] = np.nan

                break
    
    ###############dog_filter
    for i in dog:
        for j in i:
            if j not in alfa:
                class_dog[dog.index(i)] = np.nan
                dog[dog.index(i)] = np.nan
                break

    y = class_human + class_chimp + class_dog
    seq_data = human + chimp + dog
    ######################################################################
    all_data = pd.DataFrame([y,seq_data]).transpose()
    all_data.columns = ['class','sequence']
    all_data = all_data.dropna()
    all_data['class'].replace({0 : 'G protein coupled receptor', 1:'Tyrosine kinase', 2:'Tyrosine phosphatase',3:'Synthetase',4:'Synthase',5:'Ion Channel',6:'Transcription Factor'}, inplace=True)
    #iris.species.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)
    
    return all_data

def getKmers(sequence, size=5):
    kmers = []
    for x in range(len(sequence) - size + 1):
        kmers.append(sequence[x:x+size].lower())
    return kmers

def strToKmer_lstTostr(fitr_Data,X):
    
    fitr_Data['words'] = fitr_Data.apply(lambda x: getKmers(x['sequence'],X), axis=1)
    fitr_Data = fitr_Data.drop('sequence', axis=1)
    #print("X" in fitr_Data['class'])
    #print(fitr_Data.head)
    
    all_text = list(fitr_Data['words'])
    for item in range(len(all_text)):
        all_text[item] = ' '.join(all_text[item])
    #y_data = kmer_Data.iloc[:, 0].values
    y_data = fitr_Data['class']
    #print("y_data type",type(y_data))
    return all_text, y_data


def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1


def train_and_save_model(X,Y,alpha):
    
    classifier = MultinomialNB(alpha=alpha)
    classifier.fit(X,Y)
    
    with open('MultinomialNB.mod', 'wb') as m:
        pickle.dump(classifier, m)
        

    
def Accuracy(K, N ,alpha,filtr_data):

    all_texts, y_data = strToKmer_lstTostr(filtr_data,K)


    cv = CountVectorizer(ngram_range=(N,N))
    X = cv.fit_transform(all_texts)
    
    with open('CountVectorizer.mod', 'wb') as m:
        pickle.dump(cv, m)
       
    
    X_train, X_test, y_train, y_test = train_test_split(X, list(filtr_data['class']), test_size = 0.20, random_state=42)
    
    classifier = MultinomialNB(alpha=alpha)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("Confusion matrix    \n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("\n\naccuracy = %f \nprecision = %f \nrecall = %f \nf1 = %f" % (accuracy, precision, recall, f1))

    train_and_save_model(X, list(filtr_data['class']), alpha)




K = 5
N = 5
alpha = 0.2
DATA_FILTR = filter_Data(getData())

Accuracy(K, N ,alpha , DATA_FILTR)




