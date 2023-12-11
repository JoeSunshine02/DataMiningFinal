import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

##preprocess the movies.csv --> general function only run once
def preprocess():
    filepath = Path('movies_clean.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  

    movies_df = pd.read_csv("movies.csv")
    check_nan = movies_df.isnull().sum()
    print(check_nan)
    movies_df['budget'] = movies_df['budget'].fillna(0)
    movies_df['gross'] = movies_df['gross'].fillna(0)
    movies_df=movies_df.dropna()

    print(movies_df.isnull().sum())
    movies_df.to_csv(filepath,index=False)

## used to check the content of .csv
def content_check():
    movies_df = pd.read_csv("movies_clean.csv")
    print('movies_clean.csv')
    print(movies_df.head())
    print(movies_df['rating'].unique())
    print(movies_df['genre'].unique())
    print(movies_df['runtime'].max())
    #print(movies_df.isnull().sum())
    #print(movies_df['company'].value_counts())

content_check()