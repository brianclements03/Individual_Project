# INDIVIDUAL PROJECT

# TEXAS RAILROAD COMMISION DRILLING PERMIT APPROVAL TIMES

## Brian Clements
## 24 January 2022


## This file/repo contains information related to my Texas Drilling Permits project, using 2016-2021 oil and gas drilling permits for the State of Texas
## These permit databases are available for download from the Texas Railroad Commision website, http://webapps2.rrc.texas.gov/EWA/ewaMain.do

## Executive Summary

All oil and gas drilling in the State of Texas is regulated by the Texas Railroad Commission (TRRC); in the name of transparency and equitable, speedy, and efficient public service, it's important that this body be held accountable to its responsibilities.  

This study has done an exhaustive exploration of public data relating to permitting done by the TRRC, and has found that permitting is generally being done quickly across all of the TRRC's districts; in all the counties of the state; and in all the shale formations, with few exceptions.

The study has gone as far as to build a machine learning regression model in an attempt to predict the approval time for permits; while the model was able to beat the baseline prediction by a very modest 6/10ths of 1 percent, it served mainly to drive home the point that the TRRC drilling permit times are roughly the same, no matter how one looks at the data.

To put it another way: "there isn't much to see here."

## Project Description

This Jupyter Notebook and presentation explore public records relating to oil and gas permitting in the State of Texas. The data used include all of the permits that were approved from the beginning of 2016 to the end of 2021.  Apart from a general exploration of the data and potential relationships, the hope is to make a predictive model on one or more continuous variables, such as well depth, type of bore, and particularly, the amount of time a permit is approved.

I will use Residual Mean Square Error as my metric for evaluation; many models will be built using different features and hyperparameters to find the model of best fit.  One of the final deliverables will be the RMSE value resulting from my best model, contrasted with the baseline RMSE.

Additionally, a Jupyter Notebook with my main findings and conclusions will be a key deliverable; many .py files will exist as a back-up to the main Notebook (think "under-the-hood" coding that will facilitate the presentation).


## Project Planning

The intention of this project is to follow the data science pipeline by acquiring and wrangling the relevant information from the Railroad Commission website, which is availabe for download is .csv format.  Many files were downloaded and appended to one another to creat the final dataframe that the notebook is based on.  From here, a suitable cleaning/preparation was undertaken, leading to an exploration of the variables , including graphical representations and modeling.  

## Trello

The following link is for the trello board I used for the organization of this project.
https://trello.com/b/cDu1voQa

## Project Goals

The ultimate goal of this project is to build a model that predicts amount of time a permit takes to be approved; however, there is considerable doubt as to the potential predictive power of such a model, and this goal may shift as my data exploration evolves. An equally important goal, then, is a thorough exploration of the data in search of any relationship that might be found.

## Initial Questions

- Do certain leases get approval faster? Or in certain counties/districts/shales?
- What's the relation between well depth and shale/county/district?  Approval time? 
- Which district approves its permits the fastest?
- Is there a trend over time?


##  Steps to Reproduce

In  the case of this project, the first step is to download the pertinent .csv files (with the correct parameters); from here, all the required splits for the study can easily be created using the "wrangle" function in the wrangle_module.py file.  Most other visuals, statistical testing, and modelling that was done was acheived using self-defined functions that can be found in the list of .py files below.

LIST OF MODULES USED IN THE PROJECT, FOUND IN THE PROJECT DIRECTORY:
-- wrangle_module.py: for acquiring, cleaning, encoding, splitting and scaling the data.  
-- model.py: many, many different versions of the data were used in different feature selection and modeling algorithms; this module is helpful for splitting them up neatly.
-- viz.py: visuals with long code
-- hypothesis_testing: long code related to stats testing


## Data Dictionary

Variable	Meaning
___________________
- 'API_NO.' : American Petroleum Institute unique well number
- 'Operator_Name_Number' : Well operator's name and number
- 'Lease_Name' : Lease name
- 'Well' : Well number
- 'District' : District the well belongs to
- 'County' : County the well belongs to
- 'Wellbore_Profile' : Type of bore
- 'Filing_Purpose' : Purpose for filing
- 'Amend' : Was the filing amended?
- 'Total_Depth' : Total well depth
- 'Current_Queue' : Has the permit been approved? (they all have in this data)
- 'Permit_submitted' : Date the permit was submitted for approval
- 'Approval_time_days' : Number of days it took to approve the permit
- 'SHALE' : What shale formation the permit is for
- 'Depth_bin' : Bins (Deep, Middle, and Shallow) for the total well depth
- 'SHALE_BARNETT': Encoded feature for shale
- 'SHALE_EAGLE FORD' : Encoded feature for shale
- 'SHALE_HAYNESVILLE' : Encoded feature for shale
- 'SHALE_NONE' : Encoded feature for shale
- 'SHALE_PERMIAN BASIN' : Encoded feature for shale
- 'District_01' : Encoded feature for district
- 'District_02' : Encoded feature for district
- 'District_03' : Encoded feature for district
- 'District_04' : Encoded feature for district
- 'District_05' : Encoded feature for district
- 'District_06' : Encoded feature for district
- 'District_08' : Encoded feature for district
- 'District_09' : Encoded feature for district
- 'District_10' : Encoded feature for district
- 'District_7B' : Encoded feature for district
- 'District_7C' : Encoded feature for district
- 'District_8A' : Encoded feature for district
- 'Depth_scaled' : Scaled depth

Variables created in the notebook (explanation where it helps for clarity):

- permits : the main dataframe created, containing all observations
- train : a split of permits
- validate : a split of permits
- test : a split of permits
- X_train : X split of train
- y_train : y split of train
- X_validate : X split of validate
- y_validate : y split of validate
- X_test : X split of test
- y_test : y split of test
- train_scaled : train with scaled continuous attributes
- X_train_scaled : X_train with scaled attributes
- validate_scaled : validate with scaled continuous attributes
- X_validate_scaled : X_validate with scaled attributes
- test_scaled : test with scaled continuous attributes
- X_test_scaled : X_test with scaled attributes
- disc_vars : discrete variable
- temp : temporary dataframe
- quick_counties : list of counties that approve permits quickly
- slow_counties: list of counties that approve permits more slowly
- alpha : the confidence level used in stats testing
- shales : list of shale formations
- f, p = values created for statistical analysis purposes

Missing values: 

There number of rows missing vaules were in reality only a small handful, and were therefore dropped.  The final database analyzed has 67,355 rows (which is down from 76899 in an unmanipulated version of the dataframe--this includes a few outliers that were also truncated).

## Key findings, recommendations and takeaways
    
As has been demonstrated once and again, the dataset that I have built from data available on the Railroad Commision's drilling permit query tool is maddeningly limited in its predictive ability vis-a-vis the target variable. To put it another way: when it comes to oil, the State of Texas does a great job of getting permits approved quickly, efficiently, and from the looks of it, equitably.

I was able to conduct an interesting exploration of the data to get an idea of what kinds of wells are being permitted (the vast majority are new wells); which districts, shales, and counties have the highest number of permits; and, to an extent, the relationship between well depth and permitting time.  To reiterate, none of these exploratory analyses resulted in terribly unexpected outcomes, but it was interesting to get an idea of how they were interrelated, and to confirm the lack of correlation.  

To sum it up: this analysis has been able to clearly demonstrate that, based on the variable readily available in the dataset, drilling permitting is taking place quickly and evenly across jurisdictions and geographies in Texas.

## Recommendations

For the consumer of this report, the principal actionable recommendation is to continue to keep a finger on the pulse of the TRRC's work.  It seems clear from this study that this governmental body is doing a great job when it comes to drilling permits, but the public's access to information is what allows us to make sure of it.  

In the same vein, other aspects of Texas's oil output could be presented in an easier-to-analyze fashion.  As it stands, it is difficult to do an analysis of output on a very detailed level, particularly to tie new permits to their subsequent oil and gas production (summary statistics are in fact available on the TRRC website, but are broken down to a higher level than this data science study sought).  

## Next steps

This project began in the hopes of being able to link permitting and production; this goal remains elusive, due to the structure of the data available for public download at the Texas Railroad Commision website.  However, the foremost recommendation is to continue studying ways to scrape production information in order to tie it to the permitting and producer information that is so easily available, and build on the models in this report to include a predictive model for well output based on features such as shale, depth, geography, etc.  

At the very least, finding a handfull of new continuous variables to include in the anaysis could reap benefits, since the modeling in this report relies almost exclusively on categorical variables, some of which have sparse observations to model on.  This could also increase the possibility of running some feature selection and clustering.  Clustering, in particular, was not helpful in this study--apart from all the districts already being related geographically to the areas they serve (which naturally includes the shale formation, roughly), there is only the one continuous variable to attempt to utilize for a clustering model. 

- The main next step is to continue to work on building the data up from publicly available information on the Railroad Commission website, especially information on production and geography (i.e. well latitude/longitude), items that are easily downloaded but very difficult to tie to the current database.  

Apart from the above, the following is a brief list of items that I'd like to add to future iterations of the study:

- Aggregate different lease/company names where they area obscured by sub-companies
- Split and model based on time series analysis
- Remove those counties and districts with particularly few observations
- Run a feature selection algorithm


