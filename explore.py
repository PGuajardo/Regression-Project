import pandas as pd
import seaborn as sns
import scipy.stats as stats
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import acquire



'''
Creates pair plot of all the data in DataFrame
'''
def plot_variable_pairs(df):  
    #-----------------
    g = sns.PairGrid(df)
    # we can specify any two functions we want for visualization
    g.map_diag(plt.hist) # single variable
    g.map_offdiag(sns.regplot, scatter_kws={"color": "dodgerblue"}, line_kws={"color": "orange"}) # interaction of two variables



'''
Creates Lmplot, jointplot, and relplot, modified for zillow data

Needs a dataframe
Needs a ur x_var (ur target vairable) EX: 'score' or 'taxamount'
Needs a list of feautures from data set you are using (NO CATEGORICALS) EX: df_features = df[['age', 'money', 'height']]
Outputs 3 charts for each feature
'''
def plot_categorical_and_continuous_vars(df, x_var, list_of_features):
    #the name of the columns that hold the continuous and categorical features and outputs 3 different plots 
    #for visualizing a categorical variable and a continuous variable
    
    for i, col in enumerate(list_of_features):
        # i starts at 0, but plot should start at 1
        #plot_number = i + 1 

        # Create subplot.
        #plt.subplot(1, len(con_cat_features), plot_number)

        # Title with column name.
        #plt.title(col)

        # Scatter Plot
        sns.lmplot(x= x_var, y= col, data=df, line_kws={'color': 'red'})
        
        # Joint Plot
        sns.jointplot(x= x_var, y= col , data = df, kind='reg')
        
        sns.relplot(x = x_var, y = col, data = df, kind='scatter')
        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
    



'''
converts year_built to years (AKA age)
'''
# For zillow data set to create the number of years passed

def yearbuilt_years(df):
    df.year_built =  df.year_built.astype(int)
    year = date.today().year
    df['age'] = year - df.year_built
    # dropping the 'yearbuilt' column now that i have the age
    df = df.drop(columns=['year_built'])
    
    return df


'''
Converts month to year
'''
def months_to_years(df, column):
    
    df[column] = df[column] / 12
    
    return df