import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

## clean the the movies_clean.csv for the decison trees. 
# takes no paramaters
# return cleaned df.
def clean_for_tree():
    movies_df = pd.read_csv("movies_clean.csv")
    movies_df=movies_df.drop(columns=['name','released','director','writer','star','country','company'])

    #print(movies_df)
    return movies_df

## builds decision tree for the genre variable
# takes df paramater
# return no value
def rating_tree(movie_df):
    movie_df = pd.get_dummies(movie_df, columns = ['genre'], drop_first=True)

    X = movie_df.drop(columns = ['rating'])
    y = movie_df['rating']

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.1 )

    clf = tree.DecisionTreeClassifier(max_depth = 7)
    clf = clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))

    fig = plt.figure(0, (24, 12))
    # Begin your code
    tree.plot_tree(clf, class_names = ['R', 'PG', 'G', 'Not Rated', 'NC-17', 'Approved',
                                        'TV-PG', 'PG-13', 'Unrated', 'X',
                                        'TV-MA', 'TV-14'], feature_names = X.columns, filled=True)
    # End your code
    plt.show()

## builds decision tree for the genre variable
# takes df paramater
# return no value
def genre_tree(movie_df):
    movie_df = pd.get_dummies(movie_df, columns = ['rating'], drop_first=True)

    X = movie_df.drop(columns = ['genre'])
    y = movie_df['genre']

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.1 )

    clf = tree.DecisionTreeClassifier(max_depth = 7)
    clf = clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    print(accuracy_score(y_test, predicted))

    fig = plt.figure(0, (24, 12))
    # Begin your code
    tree.plot_tree(clf, class_names = ['Drama', 'Adventure', 'Action', 'Comedy', 'Horror', 'Biography', 'Crime',
 'Fantasy', 'Family', 'Sci-Fi', 'Animation', 'Romance', 'Music', 'Western',
 'Thriller', 'Mystery', 'Sport', 'Musical'], feature_names = X.columns, filled=True)
    # End your code
    plt.show()

rating_tree(clean_for_tree())
genre_tree(clean_for_tree())