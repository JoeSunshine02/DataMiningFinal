import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import plotly.express as px
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering

def clean_for_clustering():
    movies_df = pd.read_csv("movies_clean.csv")
    #movies_df=movies_df.drop(columns=['name','released','director','writer','star','country','company'])

    rating = {'R':1, 'PG':2, 'G':3, 'Not Rated':4, 'NC-17':5, 'Approved':6, 'TV-PG':7, 'PG-13':8,
              'Unrated':9, 'X':10, 'TV-MA':11, 'TV-14':12}
    genre = {'Drama':1, 'Adventure':2, 'Action':3, 'Comedy':4, 'Horror':5, 'Biography':6, 
             'Crime':7, 'Fantasy':8, 'Family':9, 'Sci-Fi':10, 'Animation':11, 'Romance':12, 
             'Music':13, 'Western':14, 'Thriller':15, 'Mystery':16, 'Sport':17, 'Musical':18}
    movies_df.replace({'rating': rating}, inplace=True)
    movies_df.replace({'genre': genre}, inplace=True)
    '''corr = movies_df.corr()
    sns.heatmap(corr, annot=True)
    plt.show()'''
    return movies_df

def cluster_dataset(movies_df):
    print(movies_df)
    all_data = movies_df[['genre', 'year', 'rating', 'score', 'budget', 'gross', 'runtime']]

    pca=PCA(n_components=2)
    pca_mdl=pca.fit_transform(all_data)
    pca_df=pd.DataFrame(pca_mdl) 

    '''
    ##this crashes if you run it
        plt.figure(figsize=(10, 7))  
        plt.title('Dendrogram method ward')
        plt.xlabel('movie_data')
        plt.ylabel('Euclidean distances')
        plt.axhline(y=825, color='r', linestyle='--')
        plt.axhline(y=1575, color='r', linestyle='--')
        dend = shc.dendrogram(shc.linkage(all_data, method='ward'))
        plt.show()
    '''
    sns.set_palette("colorblind")

    agglo = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
    y_agglo = agglo.fit_predict(all_data)
    

    y_a_df = pd.DataFrame(y_agglo, columns=['Cluster (Agglomerative)'])
    new_a_df = pd.concat([movies_df, y_a_df], axis=1)

    fig = px.scatter(pd.concat([new_a_df, pca_df], axis = 1), 
                    x = 0, y = 1, color='Cluster (Agglomerative)', hover_data=['name', 'rating','genre'])
    fig.show()


cluster_dataset(clean_for_clustering())