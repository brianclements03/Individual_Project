# imports, some redundant, that i will need to run my functions
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
    # drop columns not going into the modeling dataframe
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
    # return the dataframes for calling in notebook
    return X_train_model, X_validate_model, X_test_model

def create_features(X_train_model, X_validate_model, X_test_model):
    '''
    Same function as above, but to create the kbest and rfe dfs
    '''
    X_train_kbest = X_train_model.drop(columns = ['SHALE_BARNETT', 'SHALE_HAYNESVILLE', 'SHALE_NONE', 'District_02', 'District_03',
       'District_04', 'District_05', 'District_06',
       'District_09', 'District_10', 'District_7B', 'District_7C',
       'District_8A', 'Depth_scaled'])
    X_validate_kbest = X_validate_model.drop(columns = ['SHALE_BARNETT', 'SHALE_HAYNESVILLE', 'SHALE_NONE', 'District_02', 'District_03',
        'District_04', 'District_05', 'District_06',
        'District_09', 'District_10', 'District_7B', 'District_7C',
        'District_8A', 'Depth_scaled'])
    X_test_kbest = X_test_model.drop(columns = ['SHALE_BARNETT', 'SHALE_HAYNESVILLE', 'SHALE_NONE', 'District_02', 'District_03',
        'District_04', 'District_05', 'District_06',
        'District_09', 'District_10', 'District_7B', 'District_7C',
        'District_8A', 'Depth_scaled'])
    X_train_rfe = X_train_model.drop(columns = ['SHALE_BARNETT', 'SHALE_EAGLE FORD', 'SHALE_HAYNESVILLE', 'SHALE_NONE',
       'SHALE_PERMIAN BASIN', 'District_03',
       'District_04', 'District_08',
       'District_09', 'District_10', 'District_7B', 'District_7C',
       'District_8A', 'Depth_scaled'])
    X_validate_rfe = X_validate_model.drop(columns = ['SHALE_BARNETT', 'SHALE_EAGLE FORD', 'SHALE_HAYNESVILLE', 'SHALE_NONE',
        'SHALE_PERMIAN BASIN', 'District_03',
        'District_04', 'District_08',
        'District_09', 'District_10', 'District_7B', 'District_7C',
        'District_8A', 'Depth_scaled'])
    X_test_rfe = X_test_model.drop(columns = ['SHALE_BARNETT', 'SHALE_EAGLE FORD', 'SHALE_HAYNESVILLE', 'SHALE_NONE',
        'SHALE_PERMIAN BASIN', 'District_03',
        'District_04', 'District_08',
        'District_09', 'District_10', 'District_7B', 'District_7C',
        'District_8A', 'Depth_scaled'])
    # return the results
    return X_train_kbest, X_validate_kbest, X_test_kbest, X_train_rfe, X_validate_rfe, X_test_rfe


def run_ols_model(X_train_model, y_train, X_validate_model, y_validate, metric_df, features_description):
    '''
    Function that runs the ols model on the data
    '''
    # import the sklearn function for mean squared error
    from sklearn.metrics import mean_squared_error
    # create the model object
    lm = LinearRegression()
    # fit the model to our training data.
    lm.fit(X_train_model, y_train.Approval_time_days)
    # predict train--add ols prediction column to y_train
    y_train['Approval_time_pred_lm'] = lm.predict(X_train_model)
    # evaluate: rmse of the model on train
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_lm) ** .5
    # predict validate--add ols prediction column to y_validate
    y_validate['Approval_time_pred_lm'] = lm.predict(X_validate_model)

    # evaluate: rmse of the model on validate
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_lm) ** (0.5)
    # append the results to the metric_df dataframe
    metric_df = metric_df.append({
        'model': 'OLS Regressor ' + features_description, 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)
    # return the metric_df for calling in notebook
    return metric_df





def lasso_lars(X_train_model, y_train, X_validate_model, y_validate, metric_df, features_description):
    '''
    Function that runs the lasso lars model on the data
    
    '''
    # define the lasso lars object, including alpha
    lars = LassoLars(alpha= 1)
    # fit the model to the training data.
    lars.fit(X_train_model, y_train.Approval_time_days)
    # predict train--add lasso prediction column to y_train
    y_train['Approval_time_pred_lars'] = lars.predict(X_train_model)
    # evaluate: rmse for train
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_lars) ** (1/2)
    # predict validateadd lasso prediction column to y_validate
    y_validate['Approval_time_pred_lars'] = lars.predict(X_validate_model)
    # evaluate: rmse for validate
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_lars) ** (1/2)
    # appende results to the metric_df
    metric_df = metric_df.append({
        'model': 'Lasso_alpha1 ' + features_description, 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)
    # return the metric_df for calling in notebook
    return metric_df






def tweedie(X_train_model, y_train, X_validate_model, y_validate, metric_df, features_description):
    '''
    Function that runs the tweedie model on the data
    '''
# as seen in curriculum, the power ought to be set per distribution type
# power = 0 is same as OLS
    # creat the tweedie (aka glm) object
    glm = TweedieRegressor(power=1.4, alpha=0)
    # fit the model to the training data. 
    glm.fit(X_train_model, y_train.Approval_time_days)
    # predict train--add lasso prediction column to y_train
    y_train['Approval_time_pred_glm'] = glm.predict(X_train_model)
    # evaluate: rmse on train
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_glm) ** (1/2)
    # predict validate--add lasso prediction column to y_validate
    y_validate['Approval_time_pred_glm'] = glm.predict(X_validate_model)
    # evaluate: rmse on validate
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_glm) ** (1/2)
    # append to metric_df
    metric_df = metric_df.append({
        'model': 'glm_compound ' + features_description, 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)
    # return metric_df for calling later
    return metric_df

def polynomial_regression_deg_2(X_train_model, y_train, X_validate_model, y_validate, metric_df, features_description):
    '''
    Function that runs the polynomial model on the data
    '''    
    # make the polynomial features to get a new set of features. import from sklearn. defining degress = 2
    pf = PolynomialFeatures(degree=2)
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_model)
    # transform X_validate_scaled
    X_validate_degree2 = pf.transform(X_validate_model)
    # create the model object
    lm2 = LinearRegression()
    # fit the model to the training data.
    lm2.fit(X_train_degree2, y_train.Approval_time_days)
    # predict train--add lasso prediction column to y_train
    y_train['Approval_time_pred_lm2'] = lm2.predict(X_train_degree2)
    # evaluate: rmse on train
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_lm2) ** (1/2)
    # predict validate--add lasso prediction column to y_validate
    y_validate['Approval_time_pred_lm2'] = lm2.predict(X_validate_degree2)
    # evaluate: rmse on validate
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_lm2) ** 0.5
    # append results to metric_df
    metric_df = metric_df.append({
    'model': 'quadratic_deg2 ' + features_description, 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    #return the metric_df for calling later
    return metric_df

def polynomial_regression_deg_3(X_train_model, y_train, X_validate_model, y_validate, metric_df, features_description):
    '''
    Function that runs the polynomial model on the data
    '''        
    # make the polynomial features to get a new set of features. import from sklearn. defining degress = 3
    pf = PolynomialFeatures(degree=3)
    # fit and transform X_train_scaled
    X_train_degree3 = pf.fit_transform(X_train_model)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree3 = pf.transform(X_validate_model)
    # create the model object
    lm2 = LinearRegression()
    # fit the model to the training data.
    lm2.fit(X_train_degree3, y_train.Approval_time_days)
    # predict train--add lasso prediction column to y_train
    y_train['Approval_time_pred_lm2'] = lm2.predict(X_train_degree3)
    # evaluate: rmse on train
    rmse_train = mean_squared_error(y_train.Approval_time_days, y_train.Approval_time_pred_lm2) ** (1/2)
    # predict validate--add lasso prediction column to y_validate
    y_validate['Approval_time_pred_lm2'] = lm2.predict(X_validate_degree3)
    # evaluate: rmse on validate
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_lm2) ** 0.5
    # append results to the metric_df
    metric_df = metric_df.append({
    'model': 'quadratic_deg3 ' + features_description, 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    # return result to call later
    return metric_df



def run_all_models(X_train_model, X_train_kbest, X_train_rfe, y_train, X_validate_model,X_validate_kbest, X_validate_rfe, y_validate, metric_df):
    '''
    Function that runs all the above modeling function at the same time and returns a metric dataframe for comparison
    '''
    # call each and every of the above models so as to run them in one fell swoop.  ease of reproducibility
    #OLS
    metric_df = run_ols_model(X_train_model, y_train, X_validate_model, y_validate, metric_df, 'all features')
    metric_df = run_ols_model(X_train_kbest, y_train, X_validate_kbest, y_validate, metric_df, 'k_best')
    metric_df = run_ols_model(X_train_rfe, y_train, X_validate_rfe, y_validate, metric_df, 'rfe')
    #LASSO LARS
    metric_df = lasso_lars(X_train_model, y_train, X_validate_model, y_validate, metric_df, 'all features')
    metric_df = lasso_lars(X_train_kbest, y_train, X_validate_kbest, y_validate, metric_df, 'k_best')
    metric_df = lasso_lars(X_train_rfe, y_train, X_validate_rfe, y_validate, metric_df, 'rfe')
    #TWEEDIE
    metric_df = tweedie(X_train_model, y_train, X_validate_model, y_validate, metric_df, 'all features')
    metric_df = tweedie(X_train_kbest, y_train, X_validate_kbest, y_validate, metric_df, 'k_best')
    metric_df = tweedie(X_train_rfe, y_train, X_validate_rfe, y_validate, metric_df, 'rfe')
    #POLYNOMIAL DEG2
    metric_df = polynomial_regression_deg_2(X_train_model, y_train, X_validate_model, y_validate, metric_df, 'all features')
    metric_df = polynomial_regression_deg_2(X_train_kbest, y_train, X_validate_kbest, y_validate, metric_df, 'k_best')
    metric_df = polynomial_regression_deg_2(X_train_rfe, y_train, X_validate_rfe, y_validate, metric_df, 'rfe')
    #POLYNOMIAL DEG3
    metric_df = polynomial_regression_deg_3(X_train_model, y_train, X_validate_model, y_validate, metric_df, 'all features')
    metric_df = polynomial_regression_deg_3(X_train_kbest, y_train, X_validate_kbest, y_validate, metric_df, 'k_best')
    metric_df = polynomial_regression_deg_3(X_train_rfe, y_train, X_validate_rfe, y_validate, metric_df, 'rfe')
    # return the resulting metric_df
    return metric_df


def add_pred_mean(y_train,y_validate,y_test):
    '''
    Add baseline prediction to all y_ datasets for evaluation purposes
    '''
    # define the approval time prediction using mean
    approval_time_pred_mean = y_train.Approval_time_days.mean()
    # append this as a new column in the three y_ datasets
    y_train['Approval_time_pred_mean'] = round(approval_time_pred_mean, 6)
    y_validate['Approval_time_pred_mean'] = round(approval_time_pred_mean,6)
    y_test['Approval_time_pred_mean'] = round(approval_time_pred_mean,6)
    # return for calling in the notebook
    return y_train,y_validate,y_test


def get_rmse_in_sample(y_train,y_validate):
    '''
    Function to return a printed statement of rmse for the train and validate sets
    '''
    # import mse from sklearn library
    from sklearn.metrics import mean_squared_error
    # defing the rmse for train
    rmse_train = mean_squared_error(y_train.Approval_time_days,
                                    y_train.Approval_time_pred_mean) ** .5
    # and for validate
    rmse_validate = mean_squared_error(y_validate.Approval_time_days, y_validate.Approval_time_pred_mean) ** (0.5)
    # print statement showing the values of the above two variables
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 4), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 4))
    # return the above-defined variables for calling/assignment in the notebook
    return rmse_train, rmse_validate


def create_eval_df(rmse_train, rmse_validate):
    '''
    function to create an empty df to add evaluation metrics to as models are built
    '''
    # create a df that holds the rmse for train and validate, which were created in a previous function
    metric_df = pd.DataFrame(data=[
            {
                'model': 'mean_baseline', 
                'RMSE_train': rmse_train,
                'RMSE_validate': rmse_validate
                }
            ])
    # return the metric_df to call in notebook
    return metric_df


def create_polynomial_features_deg3(X_train_df,X_validate_df,X_test_df):
    '''
    function to create polynomial features for running model on the different X_ sets
    '''
    # take note of the variables the function is asking for: these are generic--in my notebook, the entry will be X_train_model, X_validate_model,X_test_model
    # import from sklearn to create polynomial features for running the model on test
    from sklearn.preprocessing import PolynomialFeatures
    # Make the polynomial features to get a new set of features. import from sklearn. defining degrees = 3
    pf = PolynomialFeatures(degree=3)
    # Fit and transform X_train_model (which is scaled)
    X_train_degree3 = pf.fit_transform(X_train_df)
    # Transform X_validate_model & X_test_model
    X_validate_degree3 = pf.transform(X_validate_df)
    X_test_degree3 =  pf.transform(X_test_df)
    # return the resulting dfs for calling in notebook
    return X_train_degree3, X_validate_degree3, X_test_degree3



def create_polynomial_model_object(X_test_degree3, y_test, metric_df):
    '''
    Function to create the polynomial model object, add the target prediction to the y_test dataframe, and define and print the rmse of the model on the test set
    '''
    # Create the model object
    lm2 = LinearRegression()
    # Fit the model to our test data. 
    lm2.fit(X_test_degree3, y_test.Approval_time_days)
    # Create a column in the y_test dataframe to hold the polynomial regression prediction:
    y_test['Approval_time_pred_lm2'] = lm2.predict(X_test_degree3)
    # Evaluate by calculating its RMSE on test
    rmse_test = mean_squared_error(y_test.Approval_time_days, y_test.Approval_time_pred_lm2) ** (1/2)
    # A statement reminding us of the RMSE on the train and validate samples:
    print("RMSE for Polynomial Model, degrees=3\nTraining/In-Sample: ", metric_df.RMSE_train.iloc[5], 
          "\nValidation/Out-of-Sample: ", metric_df.RMSE_validate.iloc[5],)
    # return the resulting df and variable for calling
    return y_test, rmse_test

def graph_rmse_distribution(y_test):
    # create a line representing no error
    plt.axhline(label="No Error")
    # plot approval times vs predicted approval times, with color and transparency defined
    plt.scatter(y_test.Approval_time_days, y_test.Approval_time_pred_lm2 - y_test.Approval_time_days,\
                alpha=.5, color="red", s=100, label="Model: Polynomial Regression, Deg = 3")
    # assign a legend
    plt.legend()
    # label the axes
    plt.xlabel("Actual Approval Time")
    plt.ylabel("Predicted Approval Time - Actual Approval Time")
    # give the chart a title
    plt.title("RMSE distribution on Test data--How 'wrong' the model was")
    plt.show()


def select_kbest(X,y,k):
    '''
    Function that returns top k features of a model using k best model,
    accepting X columns, y columns, and k number of top features.
    
    '''
    # K Best model here:

    from sklearn.feature_selection import SelectKBest, f_regression

    # parameters: f_regression stats test, give me 8 features
    f_selector = SelectKBest(f_regression, k=k)

    # find the top 8 X's correlated with y
    f_selector.fit(X, y)

    X_reduced = f_selector.transform(X)

    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = X.loc[:,feature_mask].columns.tolist()

    return f_feature



def select_rfe(X,y,k):
    '''
    Function that returns top k features of a model using rfe model,
    accepting X columns, y columns, and k number of top features.
    
    '''
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE

    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, k)

    # fit the data using RFE
    rfe.fit(X,y)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = X.iloc[:,feature_mask].columns.tolist()
    return rfe_feature