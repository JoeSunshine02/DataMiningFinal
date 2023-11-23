import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


movies_df = pd.read_csv("movies.csv")
check_nan = movies_df.isnull().sum()
print(check_nan)
movies_df['budget'] = movies_df['budget'].fillna(0)
movies_df['gross'] = movies_df['gross'].fillna(0)
movies_df=movies_df.dropna()

print(movies_df.isnull().sum())