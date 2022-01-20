# INDIVIDUAL PROJECT




## This file/repo contains information related to my Texas Drilling Permits project, using 2016-2021 oil and gas drilling permits for the State of Texas
## These permit databases are available for download from the Texas Railroad Commision website, http://webapps2.rrc.texas.gov/EWA/ewaMain.do

## Executive Summary



## Project Description

This Jupyter Notebook and presentation explore public records relating to oil and gas permitting in the State of Texas. The data used include all of the permits that were approved from the beginning of 2016 to the end of 2021.  Apart from a general exploration of the data and potential relationships, the hope is to make a predictive model on one or more continuous variables, such as well depth, type of bore, and particularly, the amount of time a permit is approved.

I will use Residual Mean Square Error as my metric for evaluation; many models will be built using different features and hyperparameters to find the model of best fit.  One of the final deliverables will be the RMSE value resulting from my best model, contrasted with the baseline RMSE.

Additionally, a Jupyter Notebook with my main findings and conclusions will be a key deliverable; many .py files will exist as a back-up to the main Notebook (think "under-the-hood" coding that will facilitate the presentation).


## Project Planning

The intention of this project is to follow the data science pipeline by acquiring and wrangling the relevant information from the Railroad Commission website, which is availabe for download is .csv format.  Many files were downloaded and appended to one another to creat the final dataframe that the notebook is based on.  From here, a suitable exploration of the variable will be undertaken and presented, including graphical representations and modeling.  

## Trello



## Project Goals

The ultimate goal of this project is to build a model that predicts amount of time a permit takes to be approved; however, there is considerable doubt as to the value of such a model, and this goal may shift as my data exploration evolves.

## Initial Questions

- Do certain leases get approval faster? Or in certain counties/districts/shales?
- What's the relation between well depth and shale/county/district?  Approval time? 
- Which district approves its permits the fastest?


##  Steps to Reproduce

In  the case of this project, the first step is to download the pertinent .csv files (with the correct parameters); from here, I have created several python files that can be used to aggregate, clean, prepare and otherwise manipulate the data in advance of exploration, feature selection, and modeling (listed below).

I split the data into X_train and y_train data sets for much of the exploration and modelling, and was cognizant of the independence of the target variable from other variables in the dataset.  I added a couple of features of my own, including time to approval, and dropped rows with null values (my final dataset was _____ rows long, from 52,442 that were initially aggregated from the RRC csv files)

Once the data is correctly prepared, it can be run through the sklearn preprocessing feature for polynomial regressiong and fit on the scaled X_train dataset, using only those features indicated from the recursive polynomial engineering feature selector (also an sklearn function).  This provided me with the best results for the purposes of my project.

LIST OF MODULES USED IN THE PROJECT, FOUND IN THE PROJECT DIRECTORY:
-- wrangle_module.py: for acquiring, cleaning, encoding, splitting and scaling the data.  
<!-- -- viz.py: used for creating several graphics for my final presentation
-- model.py: many, many different versions of the data were used in different feature selection and modeling algorithms; this module is helpful for splitting them up neatly.
-- feature_engineering.py: contains functions to help choose the 'best' features using certain sklearn functions  -->

## Data Dictionary

Variable	Meaning
___________________
- 

Variables created in the notebook (explanation where it helps for clarity):

-

Missing values: 

## Key findings, recommendations and takeaways
    


## Recommendations



## Next steps



The following is a brief list of items that I'd like to add to the model:

- Aggregate different lease/company names where they area obscured by sub-companies
- Continue attempts to include production information in the permits dataframe
- Pull in geographical data (lat/long) to be able to map the wells
- Different models, time series or regression (Also: Do splitting based on the the modelling done--time series or regression)
- Remove 'outlier' counties, those with a number of permits below 1.5 the IQR (only a handful of permit approvals per year/or during the whole time frame)
- Are there predictive clusters to consider? (Besides shale formation?)




