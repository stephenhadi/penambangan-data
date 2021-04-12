import sys
import nltk
from os import walk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans

#Menggunakan Agglomerative   
#Import library Agglomerative
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage



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
    #variabel untuk menyimpan hasil cluster
    c1_Agglomerative = []
    c2_Agglomerative = []
    c3_Agglomerative = []

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
        
        #kmeans_model = KMeans(n_clusters=3).fit(tfidf)       
        
        # Simpan hasil clustering berupa nomor klaster tiap objek/rekord di varialbel klaster_objek
        #klaster_objek = kmeans_model.labels_
        
         # buat model clustering dengan menggunakan jarak euclidean linkage single
        clustering_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage = 'single')
        clustering_model.fit(df)
        labels_single = clustering_model.labels_  
        
        print(labels_single)    
        for ca, doc in zip(labels_single, df):
            if ca==0:
                c1_Agglomerative.append(doc)          
                
            elif ca==1:
                 c2_Agglomerative.append(doc)
            else:
                 c3_Agglomerative.append(doc)    
                 
        #Document yang masuk ke cluster 1
        print("Document yang masuk ke cluster 1 :")
        for ca1, idx in zip(c1_Agglomerative, range(0,len(c1_Agglomerative))):
            print(idx+1,"",ca1)
            
        #Document yang masuk ke cluster 2
        print("Document yang masuk ke cluster 2 :")
        for ca2, idx in zip(c2_Agglomerative, range(0,len(c2_Agglomerative))):
            print(idx+1,"",ca2)
            
        #Document yang masuk ke cluster 3
        print("Document yang masuk ke cluster 3 :")
        for ca3, idx in zip(c3_Agglomerative, range(0,len(c3_Agglomerative))):
            print(idx+1,"",ca3)
        
    elif sys.argv[1] == "wa":  
        print("test")


if __name__ == "__main__":
    main()
