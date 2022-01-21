# the imports required for this notebook
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import wrangle_module
import model
import hypothesis_testing

def counts_w_mean_lines(df):
    '''
    Function to draw charts of the counts of all the discrete variables
    '''
    # define discrete variables for graphing purposes
    disc_vars = ['District', 'County','Wellbore_Profile', 'Filing_Purpose', 'Amend','SHALE', 'Depth_bin']
    # for loop to graph the distribution all discrete variables
    for var in disc_vars:
        plt.figure()
        # draw a seaborn countplot for every variable
        sns.countplot(x=var,data=df)
        # define the average value of the count of every variable
        mean = df[var].value_counts().mean()
        # add a horizontal line marking the average count of each variable
        plt.axhline(y=mean,label="Average")
        # rotate my x tickmarks
        plt.xticks(rotation = 45)
        # give it a title
        plt.title(label = f'Count of {var} with Mean Line')

def disrete_vs_approval_time(df):
    '''
    Function to draw up a barplot of all discrete variables against approval time
    '''
    # using the discrete variables defined above to create barplots visualizing their 
    # approval times
    disc_vars = ['District', 'County','Wellbore_Profile', 'Filing_Purpose', 'Amend','SHALE', 'Depth_bin']
    # for loop to graph the distribution all discrete variables
    for var in disc_vars:
        plt.figure()
        # draw a seaborn barplot for every variable
        sns.barplot(y= df.Approval_time_days, x=var,data=df)
        # define the average value of the count of every variable
        mean = df.Approval_time_days.mean()
        # add a horizontal line marking the  count of each variable  
        plt.axhline(y=mean,label="Average")
        # rotate my x tickmarks
        plt.xticks(rotation = 45)
        # give it a title
        plt.title(label = f'{var} against Approval Time with Mean')
        # tighten up the layout
        plt.tight_layout()

def tsa_resampling(df):
    '''
    Function to plot several resamples of the target variable over time
    '''
    #create a time-series only df for the target
    y = df.Approval_time_days
    # resample to different intervals and plot
    y.plot(alpha=.2, label='Daily')
    y.resample('W').mean().plot(alpha=.8, label='Weekly')
    y.resample('M').mean().plot(label='Monthly')
    y.resample('Y').mean().plot(label='Yearly')
    # add a legend to the plot
    plt.legend()


def shale_vs_depth(df):
    '''
    Function to plot the depth of different shale formations
    '''
    # create a matplotlib figure
    plt.figure(figsize=(4,3))
    # draw a seaborn barplot
    sns.barplot(data=df,x='SHALE',y='Total_Depth')
    # define average depth
    mean = df.Total_Depth.mean()
    # add horizontal line at the average depth
    plt.axhline(y=mean, label='average depth')
    # rotate x ticks
    plt.xticks(rotation=45)
    # add legend outside graph area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # show all
    plt.show


def approval_time_by_district(df):
    '''
    function to draw a seaborn barplot of the approval times by district
    '''
    # draw seaborn barplot
    sns.barplot(data=df,x='District',y='Approval_time_days')
    # define average approval time
    mean = df.Approval_time_days.mean()
    # add horizontal line at average approval time
    plt.axhline(y=mean, label='average approval time')
    # rotate xticks
    plt.xticks(rotation=45)
    # add legend outside of graph area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show