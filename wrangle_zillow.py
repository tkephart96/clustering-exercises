'''
Wrangle Zillow Data

Acquire/Prep:
- get_zillow
- get_object_cols
- get_numeric_cols
- nulls_by_col
- nulls_by_row
- wrangle_zillow
    - get_zillow
    - prep_zillow
        - data_prep
            - remove_columns
            - handle_missing_values

Outliers:
- clean_outliers_iqr
    - add_outlier_columns
        - get_upper_outliers
        - get_lower_outliers
- clean_outliers_qtl

Split:
- split_data

Scalers:
- mm
- std
- robs

Encode:
- encode
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
    # map county names
    df.fips = df.fips.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    df = df.rename(columns=({'fips':'county'}))
    # Derive column 'rawcensustractandblock_full' from column: 'rawcensustractandblock' make string and full length
    df.insert(11, 'rtb_full', df.apply(lambda row : str(row['rawcensustractandblock']) + ((18 - len(str(row['rawcensustractandblock']))) * '0'), axis=1))
    # df.insert(12, 'rawcensustractandblock_county', df.apply(lambda row : row['rawcensustractandblock_full'][:4], axis=1)) # this is fips as well
    df.insert(13, 'rtb_tract', df.apply(lambda row : row['rtb_full'][4:11], axis=1)) # useful for loc
    # df.insert(14, 'rtb_block', df.apply(lambda row : row['rtb_full'][11:15], axis=1)) # no good for loc
    # df.insert(15, 'rtb_extra', df.apply(lambda row : row['rtb_full'][15:], axis=1)) # no good for loc
    # remove incorrect zip codes
    df = df[df['regionidzip'] < 100000]
    # Drop rows with missing data across all columns
    df = df.dropna()
    # Replace all instances of less than bed n bath with bed n bath in column: 'roomcnt'
    df.loc[df['roomcnt'] < (df['bathroomcnt'] + df['bedroomcnt']), 'roomcnt'] = (df['bedroomcnt'] + round(df['bathroomcnt'],0))
    # now drop real 0 roomcnt
    df = df[df['roomcnt'] > 0]
    # create features
    df.insert(20, 'trx_date', df.apply(lambda row : row['transactiondate'][5:], axis=1))
    df.insert(21, 'trx_month', df.apply(lambda row : row['trx_date'][:2], axis=1))
    df = df.assign(age=(2017-df['yearbuilt']))
    df = df.assign(old_home=(df['age']>29))
    df = df.assign(has_garage=(df['garagecarcnt']>0))
    # re-type columns
    df = df.astype({'regionidzip': 'int64',
                    'rtb_tract': 'float64',
                    'trx_month': 'int64'})
    # return in order that is easier to see
    return df[
        [
            'yearbuilt',
            'age',
            'old_home',
            'bathroomcnt',
            'bedroomcnt',
            'roomcnt',
            'garagecarcnt',
            'has_garage',
            'calculatedfinishedsquarefeet',
            'latitude',
            'longitude',
            'county',
            'regionidzip',
            'rtb_tract',
            'propertylandusedesc',
            'airconditioningdesc',
            'heatingorsystemdesc',
            'taxvaluedollarcnt',
            'structuretaxvaluedollarcnt',
            'landtaxvaluedollarcnt',
            'taxamount',
            'logerror',
            'trx_month',
            'trx_date',
            'transactiondate'
        ]
    ]

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

### OUTLIERS ###

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either False (if the point is not an outlier), or 
    True if greater than the upper bound of the observation.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x:(x > upper_bound))

def get_lower_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the lower outliers for the
    series.

    The values returned will be either False (if the point is not an outlier), or 
    True if smaller than the lower bound of the observation.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: (x < lower_bound))

def add_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    for col in df.select_dtypes('number'):
        # df[f'{col}_up_outliers'] = get_upper_outliers(df[col], k)
        # df[f'{col}_lo_outliers'] = get_lower_outliers(df[col], k)
        df[f'{col}_outliers'] = get_lower_outliers(df[col], k) + get_upper_outliers(df[col], k)
    return df

def clean_outliers_iqr(df):
    '''
    Take care of outliers using IQR
    '''
    df = add_outlier_columns(df, k=1.5)
    # Drop column: 'latitude_outliers'
    df = df.drop(columns=['latitude_outliers','longitude_outliers','regionidzip_outliers','rtb_tract_outliers','rtb_block_outliers','rtb_extra_outliers'])
    # Filter rows based on column: 'roomcnt_outliers'
    df = df[df['roomcnt_outliers'] == False]
    # Filter rows based on column: 'logerror_outliers'
    df = df[df['logerror_outliers'] == False]
    # Filter rows based on column: 'structuretaxvaluedollarcnt_outliers'
    df = df[df['structuretaxvaluedollarcnt_outliers'] == False]
    # Filter rows based on column: 'landtaxvaluedollarcnt_outliers'
    df = df[df['landtaxvaluedollarcnt_outliers'] == False]
    # Filter rows based on column: 'calculatedfinishedsquarefeet_outliers'
    df = df[df['calculatedfinishedsquarefeet_outliers'] == False]
    # Filter rows based on column: 'taxamount_outliers'
    df = df[df['taxamount_outliers'] == False]
    # Filter rows based on column: 'bathroomcnt_outliers'
    df = df[df['bathroomcnt_outliers'] == False]
    # Filter rows based on column: 'yearbuilt_outliers'
    df = df[df['yearbuilt_outliers'] == False]
    # Filter rows based on column: 'taxvaluedollarcnt_outliers'
    df = df[df['taxvaluedollarcnt_outliers'] == False]
    # Filter rows based on column: 'bedroomcnt_outliers'
    df = df[df['bedroomcnt_outliers'] == False]
    # Filter rows based on column: 'garagecarcnt_outliers'
    df = df[df['garagecarcnt_outliers'] == False]
    # Drop columns
    outlier_cols = [col for col in df if col.endswith('_outliers')]
    for col in outlier_cols:
        df = df.drop(columns=col)
    return df

def clean_outliers_qtl(df):
    '''
    filter outliers based on quantile until kurtosis between +-(7) and skew between +-(2)
    used data wrangler to help with this
    '''
    # Filter rows based on column: 'logerror'
    df = df[(df['logerror'] < df['logerror'].quantile(.99)) & (df['logerror'] > df['logerror'].quantile(.005))]
    # Filter rows based on column: 'taxvaluedollarcnt'
    df = df[df['taxvaluedollarcnt'] < df['taxvaluedollarcnt'].quantile(.98)]
    # Filter rows based on column: 'structuretaxvaluedollarcnt'
    df = df[df['structuretaxvaluedollarcnt'] < df['structuretaxvaluedollarcnt'].quantile(.9965)]
    return df

### SPLIT DATA ###

def split_data(df):
    '''Split into train, validate, test with a 60/20/20 ratio'''
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=42)
    return train, validate, test

### SCALERS ###

def mm(train,validate,test,scale=None):
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
    tr = train.copy()
    tr[scale] = pd.DataFrame(mm_scale.fit_transform(tr[scale]),tr.index,tr[scale].columns)
    for col in scale:
        tr = tr.rename(columns={col: f'{col}_s'})
    if validate is not None:
        v = validate.copy()
        v[scale] = pd.DataFrame(mm_scale.transform(v[scale]),v.index,v[scale].columns)
        for col in scale:
            v = v.rename(columns={col: f'{col}_s'})
    if test is not None:
        t = test.copy()
        t[scale] = pd.DataFrame(mm_scale.transform(t[scale]),t.index,t[scale].columns)
        for col in scale:
            t = t.rename(columns={col: f'{col}_s'})
    return tr, v, t

def std(train,validate,test,scale=None):
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
    tr = train.copy()
    tr[scale] = pd.DataFrame(std_scale.fit_transform(tr[scale]),tr.index,tr[scale].columns)
    for col in scale:
        tr = tr.rename(columns={col: f'{col}_s'})
    if validate is not None:
        v = validate.copy()
        v[scale] = pd.DataFrame(std_scale.transform(v[scale]),v.index,v[scale].columns)
        for col in scale:
            v = v.rename(columns={col: f'{col}_s'})
    if test is not None:
        t = test.copy()
        t[scale] = pd.DataFrame(std_scale.transform(t[scale]),t.index,t[scale].columns)
        for col in scale:
            t = t.rename(columns={col: f'{col}_s'})
    return tr, v, t

def robs(train,validate=None,test=None,scale=None):
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
    tr = train.copy()
    tr[scale] = pd.DataFrame(rob_scale.fit_transform(tr[scale]),tr.index,tr[scale].columns)
    for col in scale:
        tr = tr.rename(columns={col: f'{col}_s'})
    if validate is not None:
        v = validate.copy()
        v[scale] = pd.DataFrame(rob_scale.transform(v[scale]),v.index,v[scale].columns)
        for col in scale:
            v = v.rename(columns={col: f'{col}_s'})
    if test is not None:
        t = test.copy()
        t[scale] = pd.DataFrame(rob_scale.transform(t[scale]),t.index,t[scale].columns)
        for col in scale:
            t = t.rename(columns={col: f'{col}_s'})
    return tr, v, t

### ENCODE ###

def encode(df,drop_first=False):
    '''
    Encode category columns from zillow dataset
    '''
    dummy = pd.get_dummies(df.select_dtypes('object'),drop_first=drop_first)
    df = pd.concat([df,dummy],axis=1)
    return df
