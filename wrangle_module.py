from itertools import dropwhile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing




def acquire_permits():
    permits_2016 = pd.read_csv('DrillingPermitResults_2016.csv',sep=',')
    permits_2017_1 = pd.read_csv('DrillingPermitResults_2017_1.csv', sep=',')
    permits_2017_2 = pd.read_csv('DrillingPermitResults_2017_2.csv', sep=',')
    permits_2018_1 = pd.read_csv('DrillingPermitResults_2018_1.csv', sep=',')
    permits_2018_2 = pd.read_csv('DrillingPermitResults_2018_2.csv', sep=',')
    permits_2019_1 = pd.read_csv('DrillingPermitResults_2019_1.csv', sep=',')
    permits_2019_2 = pd.read_csv('DrillingPermitResults_2019_2.csv', sep=',')
    permits_2020 = pd.read_csv('DrillingPermitResults_2020.csv',sep=',')
    permits_2021_1 = pd.read_csv('DrillingPermitResults_2021_1.csv', sep=',')
    permits_2021_2 = pd.read_csv('DrillingPermitResults_2021_2.csv', sep=',')
    permits = permits_2016.append(permits_2017_1).append(permits_2017_2)\
                .append(permits_2018_1).append(permits_2018_2).append(permits_2019_1)\
                .append(permits_2019_2).append(permits_2020).append(permits_2021_1)\
                .append(permits_2021_2)
    return permits



def remove_outliers(df, k, col_list):
    ''' 
    
    Here, we remove outliers from a list of columns in a dataframe and return that dataframe
    
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def prep_permits(df):
    df = df.rename(columns = {'Status Date':'Status_Date',"Status #":"Status","API NO.":'API_NO.','Operator Name/Number':'Operator_Name_Number','Lease Name':'Lease_Name','Well #':'Well','Dist.':'District','Wellbore Profile':'Wellbore_Profile','Filing Purpose':'Filing_Purpose','Total Depth':'Total_Depth','Stacked Lateral Parent Well DP #':'Stacked_Lateral_Parent_Well_DP','Current Queue':'Current_Queue'})
    x = df["Status_Date"].str.replace("Submitted", "").str.replace("Approved", "")
    x = x.str.split(n=2, expand=True)
    df["Permit_submitted"] = x[0]
    df["Permit_approved"] = x[1]
    df.Permit_submitted = pd.to_datetime(df.Permit_submitted)
    df.Permit_approved = pd.to_datetime(df.Permit_approved)
    df['Approval_time_days'] = (df.Permit_approved - df.Permit_submitted).astype(str)
    x = df['Approval_time_days'].str.split(n=2, expand=True)
    df["Approval_time_days"] = x[0].astype(int)
    shales = pd.read_excel('tx_shales_and_counties.xlsx')
    df = df.merge(shales, how='left', left_on='County', right_on='COUNTY')
    df = df.drop(columns = 'COUNTY')
    df = df.rename(columns = {'SHALE PLAY':'SHALE'})
    df = df.set_index('Permit_approved').sort_index()
    df = df.drop(columns = 'Stacked_Lateral_Parent_Well_DP')
    df['Depth_bin']=pd.qcut(df.Total_Depth,3,labels=['Shallow','Mid_depth','Deep'])
    df = remove_outliers(df,1.5,['Approval_time_days'])
    df = df.dropna()

    return df






def split_permits(df):
    '''
    Takes in the permits dataframe and returns train, validate, test subset dataframes
    '''
    # SPLIT
    # Test set is .2 of original dataframe
    train, test = train_test_split(df, test_size = .2, random_state=123)
    # The remainder is here divided .7 to train and .3 to validate
    train, validate = train_test_split(train, test_size=.3, random_state=123)

    # return train, validate, test

    X_train = train.drop(columns=['Approval_time_days'])
    y_train = pd.DataFrame(train.Approval_time_days, columns=['Approval_time_days'])

    X_validate = validate.drop(columns=['Approval_time_days'])
    y_validate = pd.DataFrame(validate.Approval_time_days, columns=['Approval_time_days'])

    X_test = test.drop(columns=['Approval_time_days'])
    y_test = pd.DataFrame(test.Approval_time_days, columns=['Approval_time_days'])

    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def encode_permits(df):
    '''
    This is encoding a few of the permit columns for later modelling; it drops the original column 
    once it has been encoded
    
    '''
    # ordinal encoder? sklearn.OrdinalEncoder

    cols_to_dummy = ['SHALE','District']
    dummy_df = pd.get_dummies(df[cols_to_dummy], dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis = 1)
    #df.columns = df.columns.astype(str)
    # I ended up renaming counties in an above function; the other encoded cols are renamed here:
    #df.rename(columns={'6037.0':'LA', '6059.0': 'Orange', '6111.0':'Ventura'}, inplace=True)
    # I have commented out the following code bc i think i might want to have the county column for exploration
    #df = df.drop(columns='county')
    return df

def scale_permits(train, validate, test):
    '''
    Takes in the permits dataframe and returns SCALED train, validate, test subset dataframes
    '''
    # SCALE
    # 1. create the object
    scaler = sklearn.preprocessing.MinMaxScaler()
    # 2. fit the object
    scaler.fit(train[['Total_Depth']])
    # 3. use the object. Scale all columns for now
    # train_scaled_df =  scaler.transform(train[['Total_Depth', 'Approval_time_days']])
    # train_scaled_df = pd.DataFrame(train_scaled_df, columns=['Total_Depth', 'Approval_time_days'])
    # train_scaled = pd.concat([train,train_scaled_df], axis = 0)

    train['Depth_scaled'] = scaler.transform(train[['Total_Depth']])
    # train['Approval_time_scaled'] = scaler.transform(train[['Approval_time_days']])
    train_scaled = train.drop(columns = ['Total_Depth'])

    validate['Depth_scaled'] = scaler.transform(validate[['Total_Depth']])
    # train['Approval_time_scaled'] = scaler.transform(train[['Approval_time_days']])
    validate_scaled = validate.drop(columns = ['Total_Depth'])

    test['Depth_scaled'] = scaler.transform(test[['Total_Depth']])
    # train['Approval_time_scaled'] = scaler.transform(train[['Approval_time_days']])
    test_scaled = test.drop(columns = ['Total_Depth'])

    # validate_scaled =  scaler.transform(validate[['Total_Depth', 'Approval_time_days']])
    # validate_scaled = pd.DataFrame(validate_scaled, columns=['Total_Depth', 'Approval_time_days'])

    # test_scaled =  scaler.transform(test[['Total_Depth', 'Approval_time_days']])
    # test_scaled = pd.DataFrame(test_scaled, columns=['Total_Depth', 'Approval_time_days'])

    # # 4. Divide into x/y

    X_train_scaled = train_scaled.drop(columns=['Approval_time_days'])
    y_train_scaled = pd.DataFrame(train_scaled.Approval_time_days, columns=['Approval_time_days'])

    X_validate_scaled = validate_scaled.drop(columns=['Approval_time_days'])
    y_validate_scaled = pd.DataFrame(validate_scaled.Approval_time_days, columns=['Approval_time_days'])

    X_test_scaled = test_scaled.drop(columns=['Approval_time_days'])
    y_test_scaled = pd.DataFrame(test_scaled.Approval_time_days, columns=['Approval_time_days'])

    return train_scaled, X_train_scaled, y_train_scaled, validate_scaled, X_validate_scaled, y_validate_scaled, test_scaled, X_test_scaled, y_test_scaled


def wrangle_df():
    # write code to check for an existing csv first
    df = acquire_permits()
    df = prep_permits(df)
    # 
    # df = remove_outliers(df, 1.5, ['Approval_time_days'])
    # 
    train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = split_permits(df)
    # 
    df = encode_permits(df)
    # 
    df = scale_permits(train,validate,test)
    # 
    train_scaled, X_train_scaled, y_train_scaled, validate_scaled, X_validate_scaled, y_validate_scaled, test_scaled, X_test_scaled, y_test_scaled = scale_permits(train,validate,test)
    # df.to_csv(df)
    return df, train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test, train_scaled, X_train_scaled, y_train_scaled, validate_scaled, X_validate_scaled, y_validate_scaled, test_scaled, X_test_scaled, y_test_scaled