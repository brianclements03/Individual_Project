# the imports required for this notebook
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import scipy.stats as stats
import sklearn.linear_model
import sklearn.preprocessing
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
# plotting defaults
plt.rc('figure', figsize=(13, 7))
plt.style.use('seaborn-whitegrid')
plt.rc('font', size=16)

import wrangle_module
import model
import viz


def levene1(df):
    # a levene test of variance for depth by approval time
    x = stats.levene(
    df[df.Approval_time_days==0].Total_Depth,
    df[df.Approval_time_days==1].Total_Depth,
    df[df.Approval_time_days==2].Total_Depth,
    df[df.Approval_time_days==3].Total_Depth,
    df[df.Approval_time_days==4].Total_Depth,
    df[df.Approval_time_days==5].Total_Depth,
    df[df.Approval_time_days==6].Total_Depth,
    df[df.Approval_time_days==7].Total_Depth,
    df[df.Approval_time_days==8].Total_Depth,
    df[df.Approval_time_days==9].Total_Depth,
    df[df.Approval_time_days==10].Total_Depth,
    df[df.Approval_time_days==11].Total_Depth,
    df[df.Approval_time_days==12].Total_Depth,
    df[df.Approval_time_days==13].Total_Depth,
    df[df.Approval_time_days==14].Total_Depth,
    df[df.Approval_time_days==15].Total_Depth,
    df[df.Approval_time_days==16].Total_Depth,
    df[df.Approval_time_days==17].Total_Depth,
    df[df.Approval_time_days==18].Total_Depth,
    df[df.Approval_time_days==19].Total_Depth,
    df[df.Approval_time_days==20].Total_Depth,
    df[df.Approval_time_days==21].Total_Depth,
    df[df.Approval_time_days==22].Total_Depth,
    df[df.Approval_time_days==23].Total_Depth,
    df[df.Approval_time_days==24].Total_Depth,
    df[df.Approval_time_days==25].Total_Depth,
    df[df.Approval_time_days==26].Total_Depth,
    df[df.Approval_time_days==27].Total_Depth,
    )
    return x

def dist_depth_by_approval_time1(df):
    ''' plot the distributions of depth by approval time, to visually check they are normally distributed'''

    for i in [df[df.Approval_time_days==0].Total_Depth,
    df[df.Approval_time_days==1].Total_Depth,
    df[df.Approval_time_days==2].Total_Depth,
    df[df.Approval_time_days==3].Total_Depth,
    df[df.Approval_time_days==4].Total_Depth,
    df[df.Approval_time_days==5].Total_Depth,
    df[df.Approval_time_days==6].Total_Depth,
    df[df.Approval_time_days==7].Total_Depth,
    df[df.Approval_time_days==8].Total_Depth,
    df[df.Approval_time_days==9].Total_Depth,
    df[df.Approval_time_days==10].Total_Depth,
    df[df.Approval_time_days==11].Total_Depth,
    df[df.Approval_time_days==12].Total_Depth,
    df[df.Approval_time_days==13].Total_Depth,
    df[df.Approval_time_days==14].Total_Depth,
    df[df.Approval_time_days==15].Total_Depth,
    df[df.Approval_time_days==16].Total_Depth,
    df[df.Approval_time_days==17].Total_Depth,
    df[df.Approval_time_days==18].Total_Depth,
    df[df.Approval_time_days==19].Total_Depth,
    df[df.Approval_time_days==20].Total_Depth,
    df[df.Approval_time_days==21].Total_Depth,
    df[df.Approval_time_days==22].Total_Depth,
    df[df.Approval_time_days==23].Total_Depth,
    df[df.Approval_time_days==24].Total_Depth,
    df[df.Approval_time_days==25].Total_Depth,
    df[df.Approval_time_days==26].Total_Depth,
    df[df.Approval_time_days==27].Total_Depth]:
        plt.figure(figsize=(4,3))
        i.hist()
        plt.show()


def kruskal_wallace1(df):
    '''function to run kw test on approval time by depth'''
    f, p = stats.kruskal(
    df[df.Approval_time_days==0].Total_Depth,
    df[df.Approval_time_days==1].Total_Depth,
    df[df.Approval_time_days==2].Total_Depth,
    df[df.Approval_time_days==3].Total_Depth,
    df[df.Approval_time_days==4].Total_Depth,
    df[df.Approval_time_days==5].Total_Depth,
    df[df.Approval_time_days==6].Total_Depth,
    df[df.Approval_time_days==7].Total_Depth,
    df[df.Approval_time_days==8].Total_Depth,
    df[df.Approval_time_days==9].Total_Depth,
    df[df.Approval_time_days==10].Total_Depth,
    df[df.Approval_time_days==11].Total_Depth,
    df[df.Approval_time_days==12].Total_Depth,
    df[df.Approval_time_days==13].Total_Depth,
    df[df.Approval_time_days==14].Total_Depth,
    df[df.Approval_time_days==15].Total_Depth,
    df[df.Approval_time_days==16].Total_Depth,
    df[df.Approval_time_days==17].Total_Depth,
    df[df.Approval_time_days==18].Total_Depth,
    df[df.Approval_time_days==19].Total_Depth,
    df[df.Approval_time_days==20].Total_Depth,
    df[df.Approval_time_days==21].Total_Depth,
    df[df.Approval_time_days==22].Total_Depth,
    df[df.Approval_time_days==23].Total_Depth,
    df[df.Approval_time_days==24].Total_Depth,
    df[df.Approval_time_days==25].Total_Depth,
    df[df.Approval_time_days==26].Total_Depth,
    df[df.Approval_time_days==27].Total_Depth
    )
    return f, p


def levene2(df):
    '''levene variance test like that run in an above cell:'''
    x = stats.levene(
    df[df.SHALE == 'PERMIAN BASIN'].Approval_time_days,
    df[df.SHALE == 'EAGLE FORD'].Approval_time_days,
    df[df.SHALE == 'NONE'].Approval_time_days,
    df[df.SHALE == 'BARNETT'].Approval_time_days,
    df[df.SHALE == 'HAYNESVILLE'].Approval_time_days
    )
    return x

def anova1(df):
    # running an anova test on the approval times by shale
    f, p =stats.f_oneway(df[df.SHALE == 'PERMIAN BASIN'].Approval_time_days,
    df[df.SHALE == 'EAGLE FORD'].Approval_time_days,
    df[df.SHALE == 'NONE'].Approval_time_days,
    df[df.SHALE == 'BARNETT'].Approval_time_days,
    df[df.SHALE == 'HAYNESVILLE'].Approval_time_days)

    return f,p


def levene3(df):
    '''levene variance test like that run in an above cell:'''
    x = stats.levene(
    df[df.District == '08'].Approval_time_days,
    df[df.District == '01'].Approval_time_days,
    df[df.District == '02'].Approval_time_days,
    df[df.District == '7C'].Approval_time_days,
    df[df.District == '8A'].Approval_time_days,
    df[df.District == '03'].Approval_time_days,
    df[df.District == '09'].Approval_time_days,
    df[df.District == '06'].Approval_time_days,
    df[df.District == '04'].Approval_time_days,
    df[df.District == '7B'].Approval_time_days,
    df[df.District == '10'].Approval_time_days,
    df[df.District == '05'].Approval_time_days
    )
    return x


def viz_district_approvals_distributions(df):
    plt.figure(figsize=(6,4))
    for i in [df[df.District == '08'].Approval_time_days,
        df[df.District == '01'].Approval_time_days,
        df[df.District == '02'].Approval_time_days,
        df[df.District == '7C'].Approval_time_days,
        df[df.District == '8A'].Approval_time_days,
        df[df.District == '03'].Approval_time_days,
        df[df.District == '09'].Approval_time_days,
        df[df.District == '06'].Approval_time_days,
        df[df.District == '04'].Approval_time_days,
        df[df.District == '7B'].Approval_time_days,
        df[df.District == '10'].Approval_time_days,
        df[df.District == '05'].Approval_time_days]:
        i.hist()

def anova2(df):
    '''running an anova test on the approval times by district'''
    f, p =stats.f_oneway(
    df[df.District == '08'].Approval_time_days,
    df[df.District == '01'].Approval_time_days,
    df[df.District == '02'].Approval_time_days,
    df[df.District == '7C'].Approval_time_days,
    df[df.District == '8A'].Approval_time_days,
    df[df.District == '03'].Approval_time_days,
    df[df.District == '09'].Approval_time_days,
    df[df.District == '06'].Approval_time_days,
    df[df.District == '04'].Approval_time_days,
    df[df.District == '7B'].Approval_time_days,
    df[df.District == '10'].Approval_time_days,
    df[df.District == '05'].Approval_time_days
    )

    return f,p