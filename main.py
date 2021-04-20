import sys
import nltk
import numpy as np
from os import walk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
import time
import string
import pickle

#Menggunakan Agglomerative   
#Import library Agglomerative
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage



def readAll():
    fileName=[]
    f={}
    for(dirpath,dirnames,filenames) in walk('./half_data'):
        fileName = filenames
    for i in fileName:
        try:
            openFile = open("data/{}".format(i),'r',encoding="utf-8", errors='surrogateescape')
            myText = openFile.read().replace("\n"," ")
            temp = myText.split('.')
            newTemp = [word.lstrip() for word in temp if len(word) > 15]
            f[i] = newTemp
        except:
            print(openFile)
    return f

def read():
    f = {}
    openFile=''
    try:
        openFile = open("30.txt", 'rb', encoding="utf-8")
        myText = openFile.read().replace("\n", " ")
        myText = myText.translate(None,string.punctuation)
        temp = myText.split('.')
        newTemp = [word.lstrip() for word in temp if len(word) > 4]
        f = newTemp
    except:
        print(openFile)
    return f

def vectorize(f):
    fileName = defaultdict(set)
    documents = []
    for key, value in f.items():
        for i in value:
            tokenText = nltk.sent_tokenize(i)

            documents = documents + tokenText
            for z in tokenText:
                fileName[z].add(key)

    tfidf = TfidfVectorizer(use_idf=False)
    result = tfidf.fit_transform(documents)
    return documents,tfidf,result,fileName


def main():

    if len(sys.argv)!= 2:
        start_time = time.time()
    elif sys.argv[1] =="basic":
        print("basic")
        start_time = time.time()
        f = readAll()
        """
        documents adalah array 1 dimensi berisi tiap kalimat pada tiap  dokumen

        tfidf adalah TfidfVectorizer

        result adalah hasil tf nya

        fileName adalah dictionary untuk mapping dengan key adalah kalimat dan valuenya adalah
        array file yang memiliki kalimat tersebut
        
        """ 
        sentence,tfidf,result,fileName = vectorize(f)
        df = pd.DataFrame(result.toarray(),columns=tfidf.get_feature_names(),index = sentence)
        #print(df)
        
        """
        Menghapus vector yang semua fitrunya bernilai 0, karena jika tidak maka 
        #saat melakukan clustering dengan affinity Cosine akan terdapat pesan error :
            "affinity cannot be used when X contains zero vectors"
        
        """
        new_df = df[~np.all(df == 0, axis=1)]
        print(new_df)


        # buat model clustering dengan menggunakan jarak euclidean linkage single
        pkl_filename = "pickle_model.pkl"
        clustering_model = ""
        if os.path.isfile(pkl_filename):
            with open(pkl_filename, 'rb') as file:
                clustering_model = pickle.load(file)
        else: 
            clustering_model = AgglomerativeClustering(distance_threshold=1, n_clusters=None,linkage = 'single', affinity = 'cosine')

            clustering_model.fit(new_df)
        
            with open(pkl_filename, 'wb') as file:
                pickle.dump(clustering_model, file)

        #Jumlah cluster
        nClusters = clustering_model.n_clusters_
        print("Jumlah cluster :", nClusters)

        #Jarak antar cluster
        distances = clustering_model.distances_
        print("Jarak antar cluster :", distances)

        #Jarak terkecil
        print("Jarak terkecil antar cluster :",distances.min())

        #Jarak terbesar
        print("Jarak terbesar antar cluster :",distances.max())

        labels_single = clustering_model.labels_
        print(labels_single)

        df_result = pd.DataFrame([])
        for ca, sentence, docs in zip(labels_single, new_df.index, fileName.values()):
            arr = fileName[sentence]
            row = pd.Series([ca, sentence, arr])
            row_df = pd.DataFrame([row])
            #Insert baris baru ke data frame
            df_result = pd.concat([row_df, df_result], ignore_index=True)

        #Rename kolom
        df_result.rename(columns = {0:'Cluster',1:'Sentence',2:'Document'}, inplace = True)
        print(df_result)

        #Menampilkan nama dokumen di setiap cluster
        for q in range(nClusters):
            df_unique = df_result[df_result['Cluster'] == q]
            #print("Cluster ",q," : \n",df_unique)
            df2 = df_unique[['Document','Sentence']]
            print("Cluster ",q," : \n",df2)

        print("program berjalan selama {:.5f} seconds".format(time.time()-start_time))
        
                     
    elif sys.argv[1] == "wa":  
        print("test")


if __name__ == "__main__":   
    main()
  