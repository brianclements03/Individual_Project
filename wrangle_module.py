from calendar import c
from itertools import dropwhile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing




def acquire_permits():
    '''
    Function to build the many drilling permit csvs into a unified dataframe
    '''
    # assigning each csv to a df with pandas 'read_csv' method
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
    # appending each permit to the last and assigning to a unified df
    permits = permits_2016.append(permits_2017_1).append(permits_2017_2)\
                .append(permits_2018_1).append(permits_2018_2).append(permits_2019_1)\
                .append(permits_2019_2).append(permits_2020).append(permits_2021_1)\
                .append(permits_2021_2)
    # return the final df
    return permits



def remove_outliers(df, k, col_list):
    ''' 
    Function to remove outliers from a list of columns in a dataframe and return that dataframe. k = multiplier for upper and lower bound definition.
    '''
    # for loop to run through the col_list entered in the function  
    for col in col_list:
        # define upper and lower quartiles
        q1, q3 = df[col].quantile([.25, .75])
        # calculate interquartile range
        iqr = q3 - q1 
        # get upper bound
        upper_bound = q3 + k * iqr   
        # get lower bound
        lower_bound = q1 - k * iqr   
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


def prep_permits(df):
    '''
    Function to clean and prep the permits df
    '''
    # rename the columns to a friendlier format
    df = df.rename(columns = {'Status Date':'Status_Date',"Status #":"Status","API NO.":'API_NO.','Operator Name/Number':'Operator_Name_Number','Lease Name':'Lease_Name','Well #':'Well','Dist.':'District','Wellbore Profile':'Wellbore_Profile','Filing Purpose':'Filing_Purpose','Total Depth':'Total_Depth','Stacked Lateral Parent Well DP #':'Stacked_Lateral_Parent_Well_DP','Current Queue':'Current_Queue'})
    # manipulate the status date column (its observations) by removing the words 'submitted' and 'approved' (replace with "")
    x = df["Status_Date"].str.replace("Submitted", "").str.replace("Approved", "")
    # split the result in two (the result was 2 dates, the submission and approval dates)
    x = x.str.split(n=2, expand=True)
    # define columns for the submission and approval dates respectively by calling their index in the x variable created above
    df["Permit_submitted"] = x[0]
    df["Permit_approved"] = x[1]
    # convert the submit and approve dates to pandas datetime format
    df.Permit_submitted = pd.to_datetime(df.Permit_submitted)
    df.Permit_approved = pd.to_datetime(df.Permit_approved)
    # add a column for the number of days for approval
    df['Approval_time_days'] = (df.Permit_approved - df.Permit_submitted).astype(str)
    # the above column was resulting in "7 days", and the following removes the string portion of each observation and defines the column as the integer only
    x = df['Approval_time_days'].str.split(n=2, expand=True)
    df["Approval_time_days"] = x[0].astype(int)
    # create a df from the below-referenced excel file
    shales = pd.read_excel('tx_shales_and_counties.xlsx')
    # merge the shales df with permits
    df = df.merge(shales, how='left', left_on='County', right_on='COUNTY')
    # drop COUNTY as a column, as it was coming up twice
    df = df.drop(columns = 'COUNTY')
    # rename the shale column
    df = df.rename(columns = {'SHALE PLAY':'SHALE'})
    # set the df index to the approved date and sort
    df = df.set_index('Permit_approved').sort_index()
    # drop a few unutilzed columns
    df = df.drop(columns = ['Stacked_Lateral_Parent_Well_DP','Status_Date', 'Status'])
    # bin the depths for study later
    df['Depth_bin']=pd.qcut(df.Total_Depth,3,labels=['Shallow','Mid_depth','Deep'])
    # run the remove outliers function
    df = remove_outliers(df,1.5,['Approval_time_days'])
    # drop the few nulls that were left
    df = df.dropna()
    # return the df for assignment in the notebook
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
    # create X_ and y_ splits by dropping the approval time column from X_
    X_train = train.drop(columns=['Approval_time_days'])
    y_train = pd.DataFrame(train.Approval_time_days, columns=['Approval_time_days'])
    # the same, for validate
    X_validate = validate.drop(columns=['Approval_time_days'])
    y_validate = pd.DataFrame(validate.Approval_time_days, columns=['Approval_time_days'])
    # and test
    X_test = test.drop(columns=['Approval_time_days'])
    y_test = pd.DataFrame(test.Approval_time_days, columns=['Approval_time_days'])
    # return the resulting dfs
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def encode_permits(df):
    '''
    This is encoding a few of the permit columns for later modelling; it drops the original column 
    once it has been encoded
    '''
    # creat a copy of the df for EDA purposes (for ease of exploration--no encoded data)
    explore = df.copy()
    # define the features to encode
    cols_to_dummy = ['SHALE','District']
    # use pandas get_dummies to create a dummy_df of the encdoded features
    dummy_df = pd.get_dummies(df[cols_to_dummy], dummy_na=False, drop_first=False)
    # concatenate the dummy_df to the original df
    df = pd.concat([df, dummy_df], axis = 1)
    # return the resulting df
    return explore, df

def scale_permits(train, validate, test):
    '''
    Takes in the permits dataframe and returns SCALED train, validate, test subset dataframes
    '''
    # SCALE
    # 1. create the object
    scaler = sklearn.preprocessing.MinMaxScaler()
    # 2. fit the object
    scaler.fit(train[['Total_Depth']])
    # create a column for the scaled depth
    train['Depth_scaled'] = scaler.transform(train[['Total_Depth']])
    # drop total depth from train_scaled, since it nows exists in a scaled version
    train_scaled = train.drop(columns = ['Total_Depth'])
    # the same process as above, for validate:
    validate['Depth_scaled'] = scaler.transform(validate[['Total_Depth']])
    # train['Approval_time_scaled'] = scaler.transform(train[['Approval_time_days']])
    validate_scaled = validate.drop(columns = ['Total_Depth'])
    # and test:
    test['Depth_scaled'] = scaler.transform(test[['Total_Depth']])
    # train['Approval_time_scaled'] = scaler.transform(train[['Approval_time_days']])
    test_scaled = test.drop(columns = ['Total_Depth'])
    # # 4. Divide into x/y
    # drop the target variable from X_train_scaled and build it into a new y_ df
    X_train_scaled = train_scaled.drop(columns=['Approval_time_days'])
    # y_train_scaled = pd.DataFrame(train_scaled.Approval_time_days, columns=['Approval_time_days'])  # NOT SCALING TARGET VARIABLES IN THIS NOTEBOOK
    # the same for validate:
    X_validate_scaled = validate_scaled.drop(columns=['Approval_time_days'])
    # y_validate_scaled = pd.DataFrame(validate_scaled.Approval_time_days, columns=['Approval_time_days'])  # NOT SCALING TARGET VARIABLES IN THIS NOTEBOOK
    # and test:
    X_test_scaled = test_scaled.drop(columns=['Approval_time_days'])
    # y_test_scaled = pd.DataFrame(test_scaled.Approval_time_days, columns=['Approval_time_days'])  # NOT SCALING TARGET VARIABLES IN THIS NOTEBOOK
    #return the variables created:
    return train_scaled, X_train_scaled, validate_scaled, X_validate_scaled, test_scaled, X_test_scaled


def wrangle_df():
    '''
    Function to run all the above in one fell swoop
    ''' 
    df = acquire_permits()
    df = prep_permits(df)
    # creates a copy of the df to feed into encoding and scaling (preserving an unencoded df for its use in the notebook)
    df1 = df.copy()
    explore, df1 = encode_permits(df1)
    train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = split_permits(df1)
    train_scaled, X_train_scaled, validate_scaled, X_validate_scaled, test_scaled, X_test_scaled = scale_permits(train,validate,test)
    # write to a new csv file for ease of moving the data as needed
    df1.to_csv('final_permits_df')
    # return the resulting dfs for assignment
    return df, df1, train, validate, test, explore, X_train, y_train, X_validate, y_validate, X_test, y_test, train_scaled, X_train_scaled, validate_scaled, X_validate_scaled, test_scaled, X_test_scaled
