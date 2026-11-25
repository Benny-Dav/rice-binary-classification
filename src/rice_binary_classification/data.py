import pandas as pd

#load data
def load_data():
    df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv?utm_source=chatgpt.com")
    return df

#inspect data
def inspect_data(df):
    #--shape of data / number of columns and rows
    print (df.shape)
    #--see all columns
    print(df.columns)
    #--see data types of all columns
    print(df.dtypes)
    #--are there any missing values?
    print(df.isna().sum()) #prints count of missing values in each column
    #--are there any duplicate records?
    print(df.duplicated().any()) #prints bool if any duplicates exist across df
    #--how many observations were made for each target class?
    print(df["Class"].value_counts()) #returns only number of unique values and the count for "Class" column
    #--percentage distribution of each class
    print(df["Class"].value_counts(normalize=True)*100)
    #--examine descriptive statistics of data
    print(df.describe())