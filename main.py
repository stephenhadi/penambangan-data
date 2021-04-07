import sys
import nltk
from os import walk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def readAll():
    fileName=[]
    f={}
    for(dirpath,dirnames,filenames) in walk('./data'):
        fileName = filenames
    for i in fileName:
        try:
            openFile = open("data/{}".format(i),'r',encoding="raw_unicode_escape")
            f[i] = openFile.read()
        except:
            print(openFile)
    return f

def read():
    f = open("data/30.txt",'r')
    return f.read()

def main():
    if len(sys.argv)!= 2:
        print("masukkan arguments basic atau wa")
    elif sys.argv[1] =="basic":
        print("basic")
        f = readAll()
        documents={}
        for key,value in f.items():
            documents[key]=nltk.sent_tokenize(value)

        tfidf = TfidfVectorizer(use_idf=False)
        result = tfidf.fit_transform(documents)
        print(result)
        df = pd.DataFrame(result.toarray(),columns=tfidf.get_feature_names(),index=f.keys())
        print(df)
    elif sys.argv[1] == "wa":
        print("test")


if __name__ == "__main__":
    main()
