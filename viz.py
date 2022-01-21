# the imports required for this notebook
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import wrangle_module
import model
import hypothesis_testing

def counts_w_mean_lines(df):
    # define discrete variables for graphing purposes
    disc_vars = ['District', 'County','Wellbore_Profile', 'Filing_Purpose', 'Amend','SHALE', 'Depth_bin']
    # for loop to graph the distribution all discrete variables
    for var in disc_vars:
        plt.figure()
        sns.countplot(x=var,data=df)
        mean = df[var].value_counts().mean()
        plt.axhline(y=mean,label="Average")
        plt.xticks(rotation = 45)
        plt.title(label = f'Count of {var} with Mean Line')

def disrete_vs_approval_time(df):
# using the discrete variables defined above to create barplots visualizing their 
# approval times
    disc_vars = ['District', 'County','Wellbore_Profile', 'Filing_Purpose', 'Amend','SHALE', 'Depth_bin']
    # for loop to graph the distribution all discrete variables
    for var in disc_vars:
        plt.figure()
        sns.barplot(y= df.Approval_time_days, x=var,data=df)
        mean = df.Approval_time_days.mean()
        plt.axhline(y=mean,label="Average")
        plt.xticks(rotation = 45)
        plt.title(label = f'{var} against Approval Time with Mean')
        plt.tight_layout()

def tsa_resampling(df):
    '''
    Function to plot several resamples of the target variable over time
    '''
    #create a time-series only df for the target
    y = df.Approval_time_days
    # resample to different intervals and plot
    y.plot(alpha=.2, label='Daily')
    # y.resample('D').mean().plot(alpha=.5, label='Daily')
    y.resample('W').mean().plot(alpha=.8, label='Weekly')
    y.resample('M').mean().plot(label='Montly')
    y.resample('Y').mean().plot(label='Yearly')
    plt.legend()


def shale_vs_depth(df):
    plt.figure(figsize=(4,3))
    sns.barplot(data=df,x='SHALE',y='Total_Depth')
    mean = df.Total_Depth.mean()
    plt.axhline(y=mean, label='average depth')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show


def approval_time_by_district(df):
    # a seaborn barplot of the approval times by district
    sns.barplot(data=df,x='District',y='Approval_time_days')
    mean = df.Approval_time_days.mean()
    plt.axhline(y=mean, label='average approval time')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show