import sys
import nltk
import numpy as np
from os import walk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import defaultdict
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
            myText = openFile.read().replace("\n"," ")
            temp = myText.split('.')
            newTemp = [word.lstrip() for word in temp if len(word) > 4]
            f[i] = newTemp
        except:
            print(openFile)
    return f

def read():
    f = open("data/30.txt",'r')
    return f.read()

def vectorize(f):
    fileName = defaultdict(set)
    documents = []
    for key, value in f.items():
        for i in value:
            tokenText = nltk.sent_tokenize(i)
            documents = documents + tokenText
            fileName[i].add(key)

    tfidf = TfidfVectorizer(use_idf=False)
    result = tfidf.fit_transform(documents)
    return documents,tfidf,result,fileName


def main():

    if len(sys.argv)!= 2:
        print("masukkan arguments basic atau wa")
    elif sys.argv[1] =="basic":
        print("basic")
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
    

        #
        # #kmeans_model = KMeans(n_clusters=3).fit(tfidf)
        #
        # # Simpan hasil clustering berupa nomor klaster tiap objek/rekord di varialbel klaster_objek
        # #klaster_objek = kmeans_model.labels_
        #
        #  # buat model clustering dengan menggunakan jarak euclidean linkage single
        # clustering_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage = 'single')
        # clustering_model.fit(df)
        # labels_single = clustering_model.labels_
        #
        # print(labels_single)
        # for ca, doc in zip(labels_single, df.index):
        #     if ca==0:
        #         c1_Agglomerative.append(doc)
        #
        #     elif ca==1:
        #          c2_Agglomerative.append(doc)
        #     else:
        #          c3_Agglomerative.append(doc)
        #
        # #Document yang masuk ke cluster 1
        # print("Document yang masuk ke cluster 1 :")
        # for ca1, idx in zip(c1_Agglomerative, range(0,len(c1_Agglomerative))):
        #     print(idx+1,"",ca1)
        #
        # #Document yang masuk ke cluster 2
        # print("Document yang masuk ke cluster 2 :")
        # for ca2, idx in zip(c2_Agglomerative, range(0,len(c2_Agglomerative))):
        #     print(idx+1,"",ca2)
        #
        # #Document yang masuk ke cluster 3
        # print("Document yang masuk ke cluster 3 :")
        # for ca3, idx in zip(c3_Agglomerative, range(0,len(c3_Agglomerative))):
        #     print(idx+1,"",ca3)
        
        # buat model clustering dengan menggunakan jarak euclidean linkage single
        clustering_model = AgglomerativeClustering(distance_threshold=1, n_clusters=None,linkage = 'single', affinity = 'cosine')
        
        clustering_model.fit(new_df)
        
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
        for ca, sentence, doc in zip(labels_single, new_df.index, fileName.values()):  
            row = pd.Series([ca, sentence, doc])
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
            df2 = df_unique['Document']
            print("Cluster ",q," : \n",df2)                
        
        
                     
    elif sys.argv[1] == "wa":  
        print("test")


if __name__ == "__main__":   
    main()
  