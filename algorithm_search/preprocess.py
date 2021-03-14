# Gets the feature and target vectors for each dataset

# imports 
import pandas as pd
import numpy as np

# get airlines features/targets
def prep_airlines():
    
    data_dir = '../data/airline_satisfaction/'
    df = pd.read_csv(data_dir + 'train.csv', index_col=0)
    df.drop('id', axis=1, inplace=True)
    df.dropna(inplace=True)

    to_encode = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    cleanup = dict()
    for col in to_encode:
        if col == 'satisfaction':
            cleanup[col] = {k:v for k,v in zip(df[col].unique(), [0,1])}
        elif len(df[col].unique()) == 2:
            cleanup[col] = {k:v for k,v in zip(df[col].unique(), [-1,1])}
        elif len(df[col].unique()) == 3:
            cleanup[col] = {k:v for k,v in zip(df[col].unique(), [-1,0,1])}

    df = df.replace(cleanup)

    X = np.array(df[df.columns[:-1]])
    y = np.array(df[df.columns[-1]])
    
    return X,y

# get incomes features/targets
def prep_income():
    
    data_dir = '../data/income/'
    df = pd.read_csv(data_dir + 'train.csv')

    drop = ['native-country']
    df.drop(drop, axis=1, inplace=True)
    df.dropna(inplace=True)

    onehot_cols = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(data=df, columns=onehot_cols)
    
    X = np.array(df[df.columns[df.columns != 'income_>50K']])
    y = np.array(df['income_>50K'])
    
    return X,y

# get phishing websites features/targets
def prep_phishing():
    
    data_dir = '../data/phishing_website/'
    df = pd.read_csv(data_dir + 'phishing.csv.xls', index_col=0)
    
    df['class'] = df['class'].map({-1:0, 1:1})
    
    X = np.array(df[df.columns[:-1]])
    y = np.array(df[df.columns[-1]])
    
    return X,y

# get surgical complications features/targets
def prep_surgical():
    
    data_dir = '../data/surgical_complications/'
    df = pd.read_csv(data_dir + 'Surgical-deepnet.csv')
    
    X = np.array(df[df.columns[:-1]])
    y = np.array(df[df.columns[-1]])
    
    return X,y