

import os, pickle
import pandas as pd



def getKmers(sequence, size=5):
    kmers = []
    for x in range(len(sequence) - size + 1):
        kmers.append(sequence[x:x+size].lower())
    return kmers




def filter_for_predict(fitr_Data,X):
    
    fitr_Data['words'] = fitr_Data.apply(lambda x: getKmers(x['sequence'],X), axis=1)
    fitr_Data = fitr_Data.drop('sequence', axis=1)

    all_text = list(fitr_Data['words'])
    for item in range(len(all_text)):
        all_text[item] = ' '.join(all_text[item])

    return all_text




def prediction(test):
    
    alphabat = set(['a','t','g','c'])
    for i in range(len(test)):
        for j in range(len(test[i])):
            test[i][j] = test[i][j].lower()
            s = set(test[i][j])
            if(s != alphabat):
                print("Ivalid! DNA Sequence \n")
                return
                
      
    df = pd.DataFrame(test, columns = ['sequence'])
    test = filter_for_predict(df, 5)
     
    with open('MultinomialNB.mod', 'rb') as m:
        classifier = pickle.load( m)
    with open('CountVectorizer.mod', 'rb') as m:
        cv = pickle.load( m)
    Test = cv.transform(test)
    
    
    print("Output class : " , classifier.predict(Test))

    




test = input("Enter Dna Sequence :  \n")
test = [[test]]
#test = [["ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCCATACTCCTTACACTATTCCTCATCACCCAACTAAAAATATTAAACACAAACTACCACCTACCTCCCTCACCAAAGCCCATAAAAATAAAAAATTATAACAAACCCTGAGAACCAAAATGAACGAAAATCTGTTCGCTTCATTCATTGCCCCCACAATCCTAG"]]
prediction(test)






