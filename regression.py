import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def clean_for_regression():
    movies_df = pd.read_csv("movies_clean.csv")
    movies_df=movies_df.drop(columns=['name','released','director','writer','star','country','company'])
    movies_df=movies_df.drop(columns=['rating', 'genre'])

    #corr = movies_df.corr()
    #sns.heatmap(corr, annot=True)
    #plt.show()
    return movies_df

def regress_budget_v_gross(movie_df): ##regression of gross vs budget
    X=movie_df['runtime'] ##OG gross
    X =np.reshape(X, (-1, 1))
    
    #X=movie_df.drop(columns=['budget'])
    y=movie_df['budget'] ## OG budget

    x_train, x_test, y_train, y_test = train_test_split(X, y)
    lr = LinearRegression() 
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    plt.scatter(x_test, y_test,  color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    plt.xlabel("runtime")
    plt.ylabel("budget")

    plt.show()

    print('Coefficients: \n', lr.coef_)
    print('Intercept: \n', lr.intercept_)

def regress_year_v_budget(movie_df): ## x=year, y=budget
    X=movie_df['year'] ##OG year
    X =np.reshape(X, (-1, 1))
    
    #X=movie_df.drop(columns=['budget'])
    y=movie_df['budget'] ## OG budget

    x_train, x_test, y_train, y_test = train_test_split(X, y)
    lr = LinearRegression() 
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    plt.scatter(x_test, y_test,  color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    print('Coefficients: \n', lr.coef_)
    print('Intercept: \n', lr.intercept_)

    
regress_budget_v_gross(clean_for_regression())