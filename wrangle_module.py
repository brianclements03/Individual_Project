import pandas as pd
import numpy as np

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



# def prep_permits(permits):
#     permits = permits.rename(columns = {'Status Date':'Status_Date',"Status #":"Status","API NO.":'API_NO.','Operator Name/Number':'Operator_Name_Number','Lease Name':'Lease_Name','Well #':'Well','Dist.':'District','Wellbore Profile':'Wellbore_Profile','Filing Purpose':'Filing_Purpose','Total Depth':'Total_Depth','Stacked Lateral Parent Well DP #':'Stacked_Lateral_Parent_Well_DP','Current Queue':'Current_Queue'})
#     x = permits["Status_Date"].str.replace("Submitted", "").str.replace("Approved", "")
#     x = x.str.split(n=2, expand=True)
#     permits["Permit_submitted"] = x[0]
#     permits["Permit_approved"] = x[1]
#     permits.Permit_submitted = pd.to_datetime(permits.Permit_submitted)
#     permits.Permit_approved = pd.to_datetime(permits.Permit_approved)
#     permits['Approval_time_days'] = (permits.Permit_approved - permits.Permit_submitted).astype(str)
#     # 
#     x = permits['Approval_time_days'].str.split(n=2, expand=True)
#     permits["Approval_time_days"] = x[0].astype(int)
#     # 
#     shales = pd.read_excel('tx_shales_and_counties.xlsx')
#     permits = permits.merge(shales, how='left', left_on='County', right_on='COUNTY')
#     permits = permits.drop(columns = 'COUNTY')
#     permits = permits.rename(columns = {'SHALE PLAY':'SHALE'})
#     permits = permits.set_index('Permit_approved').sort_index()
#     permits = permits.drop(columns = 'Stacked_Lateral_Parent_Well_DP')
#     permits = permits.dropna()

#     return permits


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
    df = df.dropna()

    return df



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


def wrangle_df():
    # write code to check for an existing csv first
    df = acquire_permits()
    df = prep_permits(df)
    # 
    df = remove_outliers(df, 1.5, ['Approval_time_days'])
    # 
    pd.to_csv(df)
    return df