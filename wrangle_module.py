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



def prep_permits(df):
    df.rename(columns = {'Status Date':'Status_Date','Status #':"Status_#" 
                           ,"API NO.":'API_NO.'
                           ,'Operator Name/Number':'Operator_Name/Number'
                           ,'Lease Name':'Lease_Name','Well #':'Well_#','Dist.':'District'
                           ,'Wellbore Profile':'Wellbore_Profile'
                           ,'Filing Purpose':'Filing_Purpose','Total Depth':'Total_Depth'
                           ,'Stacked Lateral Parent Well DP #':'Stacked_Lateral_Parent_Well_DP_#'
                           ,'Current Queue':'Current_Queue'})
    x = df["Status Date"].str.replace("Submitted", "").str.replace("Approved", "")
    x = x.str.split(n=2, expand=True)
    df["Permit_submitted"] = x[0]
    df["Permit_approved"] = x[1]

    return df