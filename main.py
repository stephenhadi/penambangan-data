import sys
import nltk
import numpy as np
from os import walk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import time
import string
import pickle
from nltk.corpus import stopwords 

#Menggunakan Agglomerative   
#Import library Agglomerative
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from matplotlib import pyplot as plt



def readAll():
    fileName=[]
    f={}
    openFile=""
    for(dirpath,dirnames,filenames) in walk('./data'):
        fileName = filenames
    for i in fileName:
        try:
            openFile = open("data/{}".format(i),'r', encoding='utf-8', errors='ignore')
            arrText = openFile.read().split("\n")
            myText = ""
            for y in arrText:
                # cari panjang diatas 9 untuk membuang kalimat kalimat yang tidak penting dan tidak memberikan bobot terlalu besar
                if len(y)> 9:
                    myText += " "+y
            f[i] = myText
        except:
            print(openFile)
    return f


def vectorize(f):
    fileName = defaultdict(list)
    documents = []
    sentenceCountPerDoc = defaultdict(int)
    stop_words = set(stopwords.words('indonesian'))
    for key, value in f.items():
        words = nltk.sent_tokenize(value.lower())
        tokenText = [word for word in words if word not in stop_words]
        documents = documents + tokenText
        for z in tokenText:
            # cek apakah nama file yang akan dimasukkan sudah ada atau belum
            # pasangan fileName {sentences:nama file}
            if key not in fileName[z]:
                fileName[z].append(key)
                sentenceCountPerDoc[key] +=1
    tfidf = TfidfVectorizer(use_idf=False,stop_words=stop_words)
    result = tfidf.fit_transform(documents)
    return documents,tfidf,result,fileName,sentenceCountPerDoc


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
        sentence,tfidf,result,fileName,sentenceCountPerDoc = vectorize(f)
        df = pd.DataFrame(result.toarray(),columns=tfidf.get_feature_names(),index = sentence)
        #print(df)

        """
        Menghapus vector yang semua fitrunya bernilai 0, karena jika tidak maka 
        #saat melakukan clustering dengan affinity Cosine akan terdapat pesan error :
            "affinity cannot be used when X contains zero vectors"
        
        """
        new_df = df[~np.all(df == 0, axis=1)]
        print(new_df)


        #buat model clustering dengan menggunakan jarak euclidean linkage single
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
            row = pd.Series([ca, sentence, docs])
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
            arr=[]
            for index, row in df_unique.iterrows():
                doc = fileName[row['Sentence']]
                arr = arr + doc
            
            arr = list(set(arr))
            res = {}
            print("Cluster ",q," : \n",arr)

        print("clustering berjalan selama {:.5f} second \n".format(time.time()-start_time))
        
        
        #cosine similarities
        start_time = time.time()
        cos_sim = cosine_similarity(new_df)
        np.fill_diagonal(cos_sim,np.nan)
        max= np.nanmax(cos_sim,1)
        max_index = np.nanargmax(cos_sim,1)
        
        result = dict()
        for i in range(len(max)):                        
            #tempSentences adalah tuple sentences yang mirip yang didapatkan berdasarkan cosine similarities diatas
            tempSentences = new_df.index[i],new_df.index[max_index[i]]
                
            documentPair = "{}|||{}".format(fileName[tempSentences[0]],fileName[tempSentences[1]])
            documentPairRev = "{}|||{}".format(fileName[tempSentences[1]],fileName[tempSentences[0]])
            #berfungi menghilangkan noise seperti PT. hlm. tegar jaya. yang dihasilkan oleh tokenize 
            if len(new_df.index[i]) >15 and fileName[tempSentences[0]] != fileName[tempSentences[1]] :
                if documentPair in  result:
                    result[documentPair] +=1
                else:
                    if documentPairRev in result:
                        result[documentPairRev] +=1
                    else:
                        result[documentPair] = 1
            
        
        i = 0
        # perbandingan berapa banyak sentences yang mirip dengan rata rata sentences kedua item
        print(sentenceCountPerDoc)
        
        for key,value in result.items():
            tempKey = key.replace("',"," '|||")
            myKey = tempKey.split("|||")
            average = 0
            for i in range(len(myKey)):
                average+= sentenceCountPerDoc.get(myKey[i][2:-2])
            average = average/len(myKey)
            result[key] = value/average
        
        sorted_result = {k.replace("|||"," "): v for k, v in sorted(result.items(), key= lambda item : item[1], reverse=True)}
        
        for key,value in sorted_result.items():
            
            print(key,value)
            i+=1
            if(i>=10):
                break
       
        
        print("cosine similarity berjalan selama {:.5f} second".format(time.time()-start_time))
        
        
        
        
    elif sys.argv[1] == "wa":  
        print("test")


if __name__ == "__main__":   
    main()
  