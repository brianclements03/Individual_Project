from this import d
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



def prep_permits(permits):
    permits = permits.rename(columns = {'Status Date':'Status_Date',"Status #":"Status","API NO.":'API_NO.','Operator Name/Number':'Operator_Name_Number','Lease Name':'Lease_Name','Well #':'Well','Dist.':'District','Wellbore Profile':'Wellbore_Profile','Filing Purpose':'Filing_Purpose','Total Depth':'Total_Depth','Stacked Lateral Parent Well DP #':'Stacked_Lateral_Parent_Well_DP','Current Queue':'Current_Queue'})
    x = permits["Status_Date"].str.replace("Submitted", "").str.replace("Approved", "")
    x = x.str.split(n=2, expand=True)
    permits["Permit_submitted"] = x[0]
    permits["Permit_approved"] = x[1]
    permits.Permit_submitted = pd.to_datetime(permits.Permit_submitted)
    permits.Permit_approved = pd.to_datetime(permits.Permit_approved)
    permits['Approval_time'] = permits.Permit_approved - permits.Permit_submitted
    shales = pd.read_excel('tx_shales_and_counties.xlsx')
    permits = permits.merge(shales, how='left', left_on='County', right_on='COUNTY')
    permits = permits.drop(columns = 'COUNTY')
    permits = permits.rename(columns = {'SHALE PLAY':'SHALE'})
    permits = permits.set_index('Permit_approved').sort_index()

    return permits




def wrangle_df():
    # write code to check for an existing csv first
    df = acquire_permits()
    df = prep_permits(df)
    pd.to_csv(df)
    return df