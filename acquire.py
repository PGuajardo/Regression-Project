import env
import pandas as pd
import os
from explore import yearbuilt_years
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#Scaling Processers
import sklearn.preprocessing

from datetime import date

#Gets connection to Code Up database using env file
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


# Get zillow.csv Data
def get_zillow_data():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        zillow = pd.read_csv(filename)
    else:
        zillow = pd.read_sql("""SELECT properties_2017.id, transactiondate, bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, fips, garagecarcnt,
garagetotalsqft, poolcnt, roomcnt, yearbuilt, taxvaluedollarcnt 
FROM predictions_2017 
JOIN properties_2017 using(parcelid) 
JOIN propertylandusetype using(propertylandusetypeid) 
WHERE propertylandusetype.propertylandusetypeid = 261 OR propertylandusetype.propertylandusetypeid = 279""", 
        get_connection('zillow'))
        zillow.to_csv(index = False)
    return zillow


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
# Remove outliers
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''

    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    df[['garagecarcnt', 'garagetotalsqft', 'poolcnt']] = df[['garagecarcnt', 'garagetotalsqft', 'poolcnt']].fillna(0)

    df = df.dropna()

    df = yearbuilt_years(df)

    df = df.drop(columns = ['transactiondate', 'yearbuilt'])

    df = df.rename(columns = {'fips':'county', 'calculatedfinishedsquarefeet' : 'area', 'bathroomcnt' : 'bathrooms',
                         'bedroomcnt' : 'bedrooms', 'poolcnt' : 'pools', 'garagecarcnt' : 'garages',
                          'taxvaluedollarcnt': 'tax_value'})

    # Create Counties By their codes
    df['LA_County']= df['county'] == 6037
    df['Orange_County']= df['county'] == 6059
    df['Ventura_County']= df['county'] == 6111

    # Set to object type
    df['county'] = df['county'].astype(object)

    # Rename County To showcase counties instead of their numeric id's
    df['county'] = df.county.replace(6059, 'Orange')
    df['county'] = df.county.replace(6037, 'LA')
    df['county'] = df.county.replace(6111, 'Ventura')

    # Replace Counties TRUE/ FALSE values with 0/1's
    df['LA_County'] = df['LA_County'].replace(False, 0)
    df['LA_County'] = df['LA_County'].replace(True, 1)

    df['Orange_County'] = df['Orange_County'].replace(False, 0)
    df['Orange_County'] = df['Orange_County'].replace(True, 1)

    df['Ventura_County'] = df['Ventura_County'].replace(False, 0)
    df['Ventura_County'] = df['Ventura_County'].replace(True, 1)

    # removing outliers
    col_list = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'garages', 'roomcnt', 'garagetotalsqft']
    k = 1.5
    
    df = remove_outliers(df, k, col_list)

    train, validate, test = split_data(df)


    return train, validate, test 

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


def split_data(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)
    return train, validate, test


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

def zillow_scaler(train, validate, test): #X_train , X_validate, X_test

    # 1. create the object
    scaler_min_max = sklearn.preprocessing.MinMaxScaler()

    # 2. fit the object (learn the min and max value)
    # train[['bedrooms', 'taxamount']]
    scaler_min_max.fit(train)

    # 3. use the object (use the min, max to do the transformation)
    # train[['bedrooms', 'taxamount']]
    scaled_bill = scaler_min_max.transform(train)

    train[['bedrooms_scaled', 'taxamount_scaled']] = scaled_bill
    # Create them on the test and validate
    # test[['bedrooms', 'taxamount']]
    test[['bedrooms_scaled', 'taxamount_scaled']] = scaler_min_max.transform(test)
    # validate[['bedrooms', 'taxamount']]
    validate[['bedrooms_scaled', 'taxamount_scaled']] = scaler_min_max.transform(validate)

    return train, validate, test #X_train, X_validate, X_test


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

def yearbuilt_years(df):
    df.yearbuilt =  df.yearbuilt.astype(int)
    year = date.today().year
    df['age'] = year - df.yearbuilt
    # dropping the 'yearbuilt' column now that i have the age
    df = df.drop(columns=['yearbuilt'])
    
    return df

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

'''
def chart_zillow(df):

    plt.figure(figsize=(16, 3))
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount', 'year_built','fips']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1
        # Create subplot.
        plt.subplot(1,9, plot_number)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        df[col].hist(bins=10, edgecolor='black')
        # Hide gridlines.
        plt.grid(False)
        plt.tight_layout()
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount', 'year_built','fips']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1
        # Create subplot.
        plt.subplot(1,9, plot_number)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        quantile_scaled_zillow2[col].hist(bins=10, edgecolor='black')
        # Hide gridlines.
        plt.grid(False)
        plt.tight_layout()
    return df
'''

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

#Do both recieve and clean of zillow data
def wrangle_zillow():

    train, validate, test = prepare_zillow(get_zillow_data())

    return train, validate, test