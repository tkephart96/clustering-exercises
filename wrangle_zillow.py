'''
Wrangle Zillow Data

Functions:
- wrangle_zillow
'''

### IMPORTS ###

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from env import user,password,host

### ACQUIRE DATA ###

def get_zillow(user=user,password=password,host=host):
    """
    This function acquires data from a SQL database of 2017 Zillow properties and caches it locally.
    
    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `get_zillow` is returning a dirty pandas DataFrame
    containing information on 2017 Zillow properties
    """
    # name of cached csv
    filename = 'zillow.csv'
    # if cached data exist
    if os.path.isfile(filename):
        df = pd.read_csv(filename, dtype={'buildingclassdesc':'string'})
    # wrangle from sql db if not cached
    else:
        # read sql query into df
        df = pd.read_sql('''
                        select *
                        from properties_2017
                        left join airconditioningtype using(airconditioningtypeid)
                        left join architecturalstyletype using(architecturalstyletypeid)
                        left join buildingclasstype using(buildingclasstypeid)
                        left join heatingorsystemtype using(heatingorsystemtypeid)
                        left join predictions_2017 using(parcelid)
                        left join propertylandusetype using(propertylandusetypeid)
                        left join storytype using(storytypeid)
                        left join typeconstructiontype using(typeconstructiontypeid)
                        where transactiondate like %s
                        '''
                        , f'mysql+pymysql://{user}:{password}@{host}/zillow'
                        , params=['2017%']
                        )
        # cache data locally
        df.to_csv(filename, index=False)
    # sort by column: 'transactiondate' (descending) for dropping dupes keeping recent
    df = df.sort_values(['transactiondate'], ascending=[False]).drop_duplicates(subset=['parcelid']).sort_index()
    # no null lat long
    df = df[df['latitude'].notna()]
    return df

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    return df.select_dtypes(exclude=['object', 'category']).columns.tolist()

def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    # pct_miss = df.isnull().sum().mean()
    pct_miss = (num_missing / df.shape[0]) * 100
    return pd.DataFrame(
        {'num_rows_missing': num_missing, 'percent_rows_missing': pct_miss}
    )

def nulls_by_row(df, index_id = 'customer_id'):
    '''
    This is a function called `nulls_by_row` that takes a pandas DataFrame `df` and 
    an optional argument `index_id` (default value is 'customer_id'). The function 
    calculates the number of missing values in each row of the DataFrame and the percentage 
    of missing values in each row. It then creates a new DataFrame `rows_missing` with 
    columns 'num_cols_missing' and 'percent_cols_missing' and merges it with the original 
    DataFrame `df` using the index. The function returns the merged DataFrame sorted by the 
    number of missing values in each row in descending order.
    '''
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': pct_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True).reset_index()[[index_id, 'num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def remove_columns(df, cols_to_remove):
    """
    This function will:
    - take in a df and list of columns
    - drop the listed columns
    - return the new df
    """
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - calculates the minimum number of non-missing values required for each column/row to be retained
    - drops columns/rows with a high proportion of missing values.
    - returns the new df
    """
    
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df

def data_prep(df, col_to_remove=None, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - list of columns
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - removes unwanted columns
    - remove rows and columns that contain a high proportion of missing values
    - returns cleaned df
    """
    if col_to_remove is None:
        col_to_remove = []
    df = remove_columns(df, col_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df

def prep_zillow(df):
    '''send uncleaned zillow df to prep and clean'''
    # Filter rows based on column: 'unitcnt' and assumed single unit
    df = df[(df['unitcnt'].isna()) | (df['unitcnt'] == 1)]
    df = df[df.propertylandusetypeid.isin([31,260,261,263,264,265,266,267,275])]
    # drop some columns and handle some major nulls
    df = data_prep(df,['parcelid','id','id.1','fullbathcnt','regionidcity','regionidcounty','finishedsquarefeet12','lotsizesquarefeet','censustractandblock','regionidneighborhood','propertyzoningdesc','assessmentyear','propertycountylandusecode','buildingqualitytypeid','calculatedbathnbr','unitcnt','garagetotalsqft'],.3,.75)
    # Replace missing values with 0 in column: 'garagecarcnt'
    df = df.fillna({'garagecarcnt': 0})
    # Replace missing values with 5 in column: 'airconditioningtypeid'
    df = df.fillna({'airconditioningtypeid': 5})
    # Replace missing values with "None" in column: 'airconditioningdesc'
    df = df.fillna({'airconditioningdesc': "None"})
    # Replace missing values with 24 in column: 'heatingorsystemtypeid'
    df = df.fillna({'heatingorsystemtypeid': 24})
    # Replace missing values with "Yes" in column: 'heatingorsystemdesc'
    df = df.fillna({'heatingorsystemdesc': "Yes"})
    # Replace missing values with taxvalue-landtax in column: 'structuretaxvaluedollarcnt'
    df = df.fillna({'structuretaxvaluedollarcnt': df['taxvaluedollarcnt']-df['landtaxvaluedollarcnt']})
    # Drop rows with missing data across all columns
    df = df.dropna()
    df.fips = df.fips.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    df = df.rename(columns=({'fips':'county'}))
    df = df.astype({'regionidzip': 'int64'})
    df = df[['yearbuilt','bathroomcnt','bedroomcnt','roomcnt','garagecarcnt','calculatedfinishedsquarefeet','latitude','longitude','county','regionidzip','rawcensustractandblock','propertylandusetypeid','propertylandusedesc','airconditioningtypeid','airconditioningdesc','heatingorsystemtypeid','heatingorsystemdesc','taxvaluedollarcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt','taxamount','logerror','transactiondate']]
    return df

def wrangle_zillow_mvp():
    """
    This function wrangles data from a SQL database of Zillow properties, caches it locally, drops null
    values, renames columns, maps county to fips, converts certain columns to integers, and handles
    outliers.
    
    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `wrangle_zillow` is returning a cleaned and wrangled pandas DataFrame
    containing information on single family residential properties in Los Angeles, Orange, and Ventura
    counties, including the year built, number of bedrooms and bathrooms, square footage, property value,
    property tax, and county. The DataFrame has been cleaned by dropping null values, renaming columns,
    mapping county codes to county names, converting certain columns
    """
    df = get_zillow()
    df = prep_zillow(df)
    return df[['beds','baths','area','prop_value']].assign(rooms=(df.beds+df.baths))

def wrangle_zillow():
    """
    This function wrangles data from a SQL database of Zillow properties, caches it locally, drops null
    values, renames columns, maps county to fips, converts certain columns to integers, and handles
    outliers.
    
    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `wrangle_zillow` is returning a cleaned and wrangled pandas DataFrame
    containing information on single family residential properties in Los Angeles, Orange, and Ventura
    counties, including the year built, number of bedrooms and bathrooms, square footage, property value,
    property tax, and county. The DataFrame has been cleaned by dropping null values, renaming columns,
    mapping county codes to county names, converting certain columns
    """
    df = get_zillow()
    df = prep_zillow(df)
    return df

### SPLIT DATA ###

def split_data(df):
    '''Split into train, validate, test with a 60/20/20 ratio'''
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=42)
    return train, validate, test

### SCALERS ###

def mm_zillow(train,validate,test,scale=None):
    """
    The function applies the Min Max Scaler method to scale the numerical features of the train, validate,
    and test datasets.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    mm_scale = MinMaxScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(mm_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(mm_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(mm_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt

def std_zillow(train,validate,test,scale=None):
    """
    The function applies the Standard Scaler method to scale the numerical features of the train, validate,
    and test datasets.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    std_scale = StandardScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(std_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(std_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(std_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt

def robs_zillow(train,validate,test,scale=None):
    """
    The function applies the RobustScaler method to scale the numerical features of the train, validate,
    and test datasets.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    rob_scale = RobustScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(rob_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(rob_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(rob_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt

### ENCODE ###

def encode_county(df):
    '''
    Encode county column from zillow dataset
    '''
    df['Orange'] = df.county.map({'Orange':1,'Ventura':0,'LA':0})
    df['LA'] = df.county.map({'Orange':0,'Ventura':0,'LA':1})
    df['Ventura'] = df.county.map({'Orange':0,'Ventura':1,'LA':0})
    return df
