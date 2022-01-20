import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydataset import data
import statistics
import seaborn as sns
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import sklearn.preprocessing
import warnings
warnings.filterwarnings("ignore")
# importing my personal wrangle module
import wrangle_module


def create_data_for_models(X_train_scaled, X_validate_scaled, X_test_scaled):
    '''
    This function takes a DataFrame and manipulates it (by dropping features) to arrive at a set of features
    to put into different models.  All features are dropped except those categoricals that are encoded and the 
    single continuous feature in the data set.
    
    '''
    X_train_model = X_train_scaled.drop(columns = ['API_NO.', 'Operator_Name_Number',
       'Lease_Name', 'Well', 'District', 'County', 'Wellbore_Profile',
       'Filing_Purpose', 'Amend', 'Current_Queue', 'Permit_submitted', 'SHALE',
       'Depth_bin'])
    X_validate_model = X_validate_scaled.drop(columns = ['API_NO.', 'Operator_Name_Number',
       'Lease_Name', 'Well', 'District', 'County', 'Wellbore_Profile',
       'Filing_Purpose', 'Amend', 'Current_Queue', 'Permit_submitted', 'SHALE',
       'Depth_bin'])
    X_test_model = X_test_scaled.drop(columns = ['API_NO.', 'Operator_Name_Number',
       'Lease_Name', 'Well', 'District', 'County', 'Wellbore_Profile',
       'Filing_Purpose', 'Amend', 'Current_Queue', 'Permit_submitted', 'SHALE',
       'Depth_bin'])


    return X_train_model, X_validate_model, X_test_model




def run_ols_model(X_train_model, y_train, X_validate_model, y_validate, metric_df):
    '''
    Function that runs the ols model on the data
    
    '''

    from sklearn.metrics import mean_squared_error
    # create the model object
    lm = LinearRegression()
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_model, y_train.Approval_time_days)
    # predict train
    y_train['Approval_time_pred_lm'] = lm.predict(X_train_model)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_lm) ** .5
    # predict validate
    y_validate['Approval_time_pred_lm'] = lm.predict(X_validate_model)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_lm) ** (0.5)
    
    metric_df = metric_df.append({
        'model': 'OLS Regressor', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)

    # print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
    #     "\nValidation/Out-of-Sample: ", rmse_validate)

    return metric_df





def lasso_lars(X_train_model, y_train, X_validate_model, y_validate, metric_df):
    '''
    Function that runs the lasso lars model on the data
    
    '''

    # a good balance is a low rmse and a low difference

    lars = LassoLars(alpha= 1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(X_train_model, y_train.Approval_time_days)

    # predict train
    y_train['Approval_time_pred_lars'] = lars.predict(X_train_model)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_lars) ** (1/2)

    # predict validate
    y_validate['Approval_time_pred_lars'] = lars.predict(X_validate_model)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_lars) ** (1/2)
    metric_df = metric_df.append({
        'model': 'Lasso_alpha1', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)
    
    return metric_df






def tweedie(X_train_model, y_train, X_validate_model, y_validate, metric_df):

    '''
    Function that runs the tweedie model on the data
    
    '''

# as seen in curriculum, the power ought to be set per distribution type
# power = 0 is same as OLS

    glm = TweedieRegressor(power=1.4, alpha=0)


    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train_model, y_train.Approval_time_days)

    # predict train
    y_train['Approval_time_pred_glm'] = glm.predict(X_train_model)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_glm) ** (1/2)

    # predict validate
    y_validate['Approval_time_pred_glm'] = glm.predict(X_validate_model)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_glm) ** (1/2)

    metric_df = metric_df.append({
        'model': 'glm_compound', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)
    return metric_df

def polynomial_regression_deg_2(X_train_model, y_train, X_validate_model, y_validate, X_test_model, metric_df):
    '''
    Function that runs the polynomial model on the data
    
    '''    
        
        # make the polynomial features to get a new set of features. import from sklearn
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_model)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_model)
    X_test_degree2 =  pf.transform(X_test_model)
    # create the model object
    lm2 = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.Approval_time_days)

    # predict train
    y_train['Approval_time_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_lm2) ** (1/2)

    # predict validate
    y_validate['Approval_time_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_lm2) ** 0.5

    metric_df = metric_df.append({
    'model': 'quadratic_deg2', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    return metric_df

def polynomial_regression_deg_3(X_train_model, y_train, X_validate_model, y_validate, X_test_model, metric_df):
    '''
    Function that runs the polynomial model on the data
    
    '''    
        
        # make the polynomial features to get a new set of features. import from sklearn
    pf = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    X_train_degree3 = pf.fit_transform(X_train_model)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree3 = pf.transform(X_validate_model)
    X_test_degree3 =  pf.transform(X_test_model)
    # create the model object
    lm2 = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree3, y_train.Approval_time_days)

    # predict train
    y_train['Approval_time_pred_lm2'] = lm2.predict(X_train_degree3)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_lm2) ** (1/2)

    # predict validate
    y_validate['Approval_time_pred_lm2'] = lm2.predict(X_validate_degree3)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_lm2) ** 0.5

    metric_df = metric_df.append({
    'model': 'quadratic_deg3', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    return metric_df



def run_all_models(X_train_model, y_train, X_validate_model, y_validate, X_test_model, metric_df):
    '''
    Function that runs all the above modeling function at the same time and returns a metric dataframe for comparison
    
    '''
    metric_df = run_ols_model(X_train_model, y_train, X_validate_model, y_validate, metric_df)
    metric_df = lasso_lars(X_train_model, y_train, X_validate_model, y_validate, metric_df)
    metric_df = tweedie(X_train_model, y_train, X_validate_model, y_validate, metric_df)
    metric_df = polynomial_regression_deg_2(X_train_model, y_train, X_validate_model, y_validate, X_test_model, metric_df)
    metric_df = polynomial_regression_deg_3(X_train_model, y_train, X_validate_model, y_validate, X_test_model, metric_df)

    return metric_df