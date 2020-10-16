import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

# This file produce the TPR disparities over 5 run plots

Run5_sex = pd.read_csv("./results77/TPR5_Disparities_sex.csv")
Run4_sex = pd.read_csv("./results19/TPR4_Disparities_sex.csv")
Run3_sex = pd.read_csv("./results38/TPR3_Disparities_sex.csv")
Run2_sex = pd.read_csv("./results47/TPR2_Disparities_sex.csv")
Run1_sex = pd.read_csv("./results31/TPR1_Disparities_sex.csv")

Run5_Age = pd.read_csv("./results77/TPR5_Disparities_Age.csv")
Run4_Age = pd.read_csv("./results19/TPR4_Disparities_Age.csv")
Run3_Age = pd.read_csv("./results38/TPR3_Disparities_Age.csv")
Run2_Age = pd.read_csv("./results47/TPR2_Disparities_Age.csv")
Run1_Age = pd.read_csv("./results31/TPR1_Disparities_Age.csv")

Run5_race = pd.read_csv("./results77/TPR5_Disparities_race.csv")
Run4_race = pd.read_csv("./results19/TPR4_Disparities_race.csv")
Run3_race = pd.read_csv("./results38/TPR3_Disparities_race.csv")
Run2_race = pd.read_csv("./results47/TPR2_Disparities_race.csv")
Run1_race = pd.read_csv("./results31/TPR1_Disparities_race.csv")

Run5_insurance = pd.read_csv("./results77/TPR5_Disparities_insurance.csv")
Run4_insurance = pd.read_csv("./results19/TPR4_Disparities_insurance.csv")
Run3_insurance = pd.read_csv("./results38/TPR3_Disparities_insurance.csv")
Run2_insurance = pd.read_csv("./results47/TPR2_Disparities_insurance.csv")
Run1_insurance = pd.read_csv("./results31/TPR1_Disparities_insurance.csv")

diseases = ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']

# diseases_abbr = {'Cardiomegaly': 'Cd',
#                 'Effusion': 'Ef',
#                 'Enlarged Cardiomediastinum': 'EC',
#                 'Lung Lesion': 'LL',
#                 'Atelectasis': 'A',
#                 'Pneumonia': 'Pa',
#                 'Pneumothorax': 'Px',
#                 'Consolidation': 'Co',
#                 'Edema': 'Ed',
#                 'Pleural Effusion': 'PE',
#                 'Pleural Other': 'PO',
#                 'Fracture': 'Fr',
#                 'Support Devices': 'SD',
#                 'Airspace Opacity': 'AO',
#                 'No Finding': 'NF'
#                 }

diseases_abbr = {'Cardiomegaly': 'Cardiomegaly',
                'Effusion': 'Effusion',
                'Enlarged Cardiomediastinum': 'Enlarged Card.',
                'Lung Lesion': 'Lung Lesion',
                'Atelectasis': 'Atelectasis',
                'Pneumonia': 'Pneumonia',
                'Pneumothorax': 'Pneumothorax',
                'Consolidation': 'Consolidation',
                'Edema': 'Edema',
                'Pleural Effusion': 'Effusion',
                'Pleural Other': 'Pleural Other',
                'Fracture': 'Fracture',
                'Support Devices': 'Sup. Devices',
                'Airspace Opacity': 'Air. Opacity',
                'No Finding': 'No Finding'
                }


m = 6
F=14
deg =15
resultX = Run1_sex.append([Run2_sex, Run3_sex,Run4_sex, Run5_sex])

resX =resultX.groupby("diseases")
resX_df = resX.describe()


dfM0 = resX_df['%M']['mean']
dfM1 = resX_df['Gap_M']["mean"]
dfM2 = 1.96 * resX_df['Gap_M']["std"] / np.sqrt(5)

dfF0 = resX_df['%F']['mean']
dfF1 = resX_df['Gap_F']["mean"]
dfF2 = 1.96 * resX_df['Gap_F']["std"] / np.sqrt(5)

Want = pd.DataFrame(pd.DataFrame(diseases, columns=["diseases"]))
PrcentM_list = []
CIM_list = []
Gap_M_mean_list = []

diseases_abbr_list = []
Distance_list = []

PrcentF_list = []
CIF_list = []
PrcentF_list = []
Gap_F_mean_list = []

for disease in diseases:
    PrcentM_list.append(dfM0[disease])
    Gap_M_mean_list.append(dfM1[disease])
    CIM_list.append(dfM2[disease])

    PrcentF_list.append(dfF0[disease])
    Gap_F_mean_list.append(dfF1[disease])
    CIF_list.append(dfF2[disease])

    Distance_list.append(np.absolute(dfF1[disease] - dfM1[disease]))
    diseases_abbr_list.append(diseases_abbr[disease])

d = {'diseases': diseases, 'diseases_abbr': diseases_abbr_list, 'Distance': Distance_list,
     "%M": PrcentM_list, 'Gap_M_mean': Gap_M_mean_list, 'CI_M': CIM_list,
     "%F": PrcentF_list, 'Gap_F_mean': Gap_F_mean_list, 'CI_F': CIF_list
     }
WantX = pd.DataFrame(d)

#Want = Want.sort_values(by ='Distance' )
WantX = WantX.sort_values(by ='Distance' )
plt.rcParams.update({'font.size': F})

plt.figure(figsize=(16,m))
plt.scatter(WantX['diseases_abbr'],WantX['Gap_M_mean'],s= np.multiply(WantX['%M'],500), marker='o',color='blue', label="Male")
plt.errorbar(WantX['diseases_abbr'],WantX['Gap_M_mean'],yerr = WantX['CI_M'],fmt='o',mfc='blue')
plt.scatter(WantX['diseases_abbr'],WantX['Gap_F_mean'],s= np.multiply(WantX['%F'],500), marker='o',color='red', label="Female")
plt.errorbar(WantX['diseases_abbr'],WantX['Gap_F_mean'],yerr = WantX['CI_F'],fmt='o',mfc='red')


plt.xticks(rotation=deg)
plt.ylabel("TPR SEX DISPARITY")
plt.legend()
plt.grid(True)
plt.savefig("./results/TPR_Dis_SEX.pdf")

#---------------------------------------------------   Age



resultAge = Run1_Age.append([Run2_Age, Run3_Age, Run4_Age, Run5_Age])
#resultAge.to_csv("./results/resultAge.csv")
resAge =resultAge.groupby("diseases")
resAge_df = resAge.describe()
#resAge_df.to_csv("./results/resAge_df.csv")

Number = 5

df40 = resAge_df['%40-60']['mean']
df41 = resAge_df['Gap_40-60']["mean"]
df42 = 1.96*resAge_df['Gap_40-60']["std"]/np.sqrt(Number)

df60 = resAge_df['%60-80']['mean']
df61 = resAge_df['Gap_60-80']["mean"]
df62 = 1.96*resAge_df['Gap_60-80']["std"]/np.sqrt(Number)

df20 = resAge_df['%20-40']['mean']
df21 = resAge_df['Gap_20-40']["mean"]
df22 = 1.96*resAge_df['Gap_20-40']["std"]/np.sqrt(Number)

df80 = resAge_df['%80-']['mean']
df81 = resAge_df['Gap_80-']["mean"]
df82 = 1.96*resAge_df['Gap_80-']["std"]/np.sqrt(Number)

df00 = resAge_df['%0-20']['mean']
df01 = resAge_df['Gap_0-20']["mean"]
df02 = 1.96*resAge_df['Gap_0-20']["std"]/np.sqrt(Number)

WantAge = pd.DataFrame(pd.DataFrame(diseases, columns=["diseases"]))
Prcent40_list = []
CI40_list = []
Gap_40_mean_list = []
diseases_abbr_list = []
Distance_list = []

Prcent60_list = []
CI60_list = []
Prcent60_list = []
Gap_60_mean_list = []

Prcent20_list = []
CI20_list = []
Prcent20_list = []
Gap_20_mean_list = []

Prcent80_list = []
CI80_list = []
Prcent80_list = []
Gap_80_mean_list = []

Prcent0_list = []
CI0_list = []
Prcent0_list = []
Gap_0_mean_list = []

Mean_list = []

for disease in diseases:
    Mean_list = []
    cleanedList_mean = []
    Prcent40_list.append(df40[disease])
    Gap_40_mean_list.append(df41[disease])
    CI40_list.append(df42[disease])
    Mean_list.append(df41[disease])

    Prcent60_list.append(df60[disease])
    Gap_60_mean_list.append(df61[disease])
    CI60_list.append(df62[disease])
    Mean_list.append(df61[disease])

    Prcent20_list.append(df20[disease])
    Gap_20_mean_list.append(df21[disease])
    CI20_list.append(df22[disease])
    Mean_list.append(df21[disease])

    Prcent80_list.append(df80[disease])
    Gap_80_mean_list.append(df81[disease])
    CI80_list.append(df82[disease])
    Mean_list.append(df81[disease])

    Prcent0_list.append(df00[disease])
    Gap_0_mean_list.append(df01[disease])
    CI0_list.append(df02[disease])
    Mean_list.append(df01[disease])


    cleanedList_mean = [x for x in Mean_list if str(x) != 'nan']
    Distance_list.append(np.max(cleanedList_mean) - np.min(cleanedList_mean))
    diseases_abbr_list.append(diseases_abbr[disease])

d = {'diseases': diseases, 'diseases_abbr': diseases_abbr_list, 'Distance': Distance_list,
     "%40-60": Prcent40_list, 'Gap_40-60_mean': Gap_40_mean_list, 'CI_40-60': CI40_list,
     "%60-80": Prcent60_list, 'Gap_60-80_mean': Gap_60_mean_list, 'CI_60-80': CI60_list,
     "%20-40": Prcent20_list, 'Gap_20-40_mean': Gap_20_mean_list, 'CI_20-40': CI20_list,
     "%80-": Prcent80_list, 'Gap_80-_mean': Gap_80_mean_list, 'CI_80-': CI80_list,
     "%0-20": Prcent0_list, 'Gap_0-20_mean': Gap_0_mean_list, 'CI_0-20': CI0_list
     }
WantAge = pd.DataFrame(d)

WantAge = WantAge.sort_values(by ='Distance' )
WantAge.to_csv("./results/WantAge.csv")


plt.rcParams.update({'font.size': F})

plt.figure(figsize=(16,m))
plt.scatter(WantAge['diseases_abbr'],WantAge['Gap_60-80_mean'],s= np.multiply(WantAge['%60-80'],500), marker='o',color='blue', label="60-80")
plt.errorbar(WantAge['diseases_abbr'],WantAge['Gap_60-80_mean'],yerr = WantAge['CI_60-80'],fmt='o',mfc='blue')
plt.scatter(WantAge['diseases_abbr'],WantAge['Gap_40-60_mean'],s= np.multiply(WantAge['%40-60'],500), marker='o',color='orange', label="40-60")
plt.errorbar(WantAge['diseases_abbr'],WantAge['Gap_40-60_mean'],yerr = WantAge['CI_40-60'],fmt='o',mfc='orange')
plt.scatter(WantAge['diseases_abbr'],WantAge['Gap_20-40_mean'],s= np.multiply(WantAge['%20-40'],500), marker='o',color='green', label="20-40")
plt.errorbar(WantAge['diseases_abbr'],WantAge['Gap_20-40_mean'],yerr = WantAge['CI_20-40'],fmt='o',mfc='green')
plt.scatter(WantAge['diseases_abbr'],WantAge['Gap_80-_mean'],s= np.multiply(WantAge['%80-'],500), marker='o',color='red', label="80-")
plt.errorbar(WantAge['diseases_abbr'],WantAge['Gap_80-_mean'],yerr = WantAge['CI_80-'],fmt='o',mfc='red')
plt.scatter(WantAge['diseases_abbr'],WantAge['Gap_0-20_mean'],s= np.multiply(WantAge['%0-20'],500), marker='o',color='purple', label="0-20")
plt.errorbar(WantAge['diseases_abbr'],WantAge['Gap_0-20_mean'],yerr = WantAge['CI_0-20'],fmt='o',mfc='purple')


plt.xticks(rotation = deg)
plt.ylabel("TPR AGE DISPARITY")
plt.legend()
plt.grid(True)
plt.savefig("./results/TPR_Dis_AGE.pdf")

#---------------------------------------------------------------------race
resultR = Run1_race.append([Run2_race, Run3_race,Run4_race, Run5_race])

#resultR.to_csv("./results/resultR.csv")


resR =resultR.groupby("diseases")


resR_df = resR.describe()

#resR_df.to_csv("./results/resR.csv")

#print(resR.columns)

WantR = pd.DataFrame(pd.DataFrame(diseases, columns=["diseases"]))

dfW0 = resR_df['%White']['mean']
dfW1 = resR_df['Gap_White']["mean"]
dfW2 = 1.96 * resR_df['Gap_White']["std"] / np.sqrt(5)


dfB0 = resR_df['%Black']['mean']
dfB1 = resR_df['Gap_Black']["mean"]
dfB2 = 1.96 * resR_df['Gap_Black']["std"] / np.sqrt(5)

dfH0 = resR_df['%Hisp']['mean']
dfH1 = resR_df['Gap_Hisp']["mean"]
dfH2 = 1.96 * resR_df['Gap_Hisp']["std"] / np.sqrt(5)

dfOt0 = resR_df['%Other']['mean']
dfOt1 = resR_df['Gap_Other']["mean"]
dfOt2 = 1.96 * resR_df['Gap_Other']["std"] / np.sqrt(5)

dfAs0 = resR_df['%Asian']['mean']
dfAs1 = resR_df['Gap_Asian']["mean"]
dfAs2 = 1.96 * resR_df['Gap_Asian']["std"] / np.sqrt(5)

dfAM0 = resR_df['%American']['mean']
dfAm1 = resR_df['Gap_American']["mean"]
dfAm2 = 1.96 * resR_df['Gap_American']["std"] / np.sqrt(5)


PrcentAs_list = []
CIAs_list = []
Gap_As_mean_list = []


CIAm_list = []
PrcentAm_list = []
Gap_Am_mean_list = []


PrcentW_list = []
CIW_list = []
Gap_W_mean_list = []

PrcentB_list = []
CIB_list = []
Gap_B_mean_list = []

PrcentH_list = []
CIH_list = []
Gap_H_mean_list = []

PrcentOt_list = []
CIOt_list = []
Gap_Ot_mean_list = []

diseases_abbr_list = []
Distance_list = []


for disease in diseases:
    Mean_list = []
    # print(disease)
    PrcentB_list.append(dfB0[disease])
    Gap_B_mean_list.append(dfB1[disease])
    CIB_list.append(dfB2[disease])
    Mean_list.append(dfB1[disease])

    PrcentH_list.append(dfH0[disease])
    Gap_H_mean_list.append(dfH1[disease])
    CIH_list.append(dfH2[disease])
    Mean_list.append(dfH1[disease])

    PrcentOt_list.append(dfOt0[disease])
    Gap_Ot_mean_list.append(dfOt1[disease])
    CIOt_list.append(dfOt2[disease])
    Mean_list.append(dfOt1[disease])

    PrcentW_list.append(dfW0[disease])
    Gap_W_mean_list.append(dfW1[disease])
    CIW_list.append(dfW2[disease])
    Mean_list.append(dfW1[disease])

    PrcentAs_list.append(dfAs0[disease])
    Gap_As_mean_list.append(dfAs1[disease])
    CIAs_list.append(dfAs2[disease])
    Mean_list.append(dfAs1[disease])

    PrcentAm_list.append(dfAM0[disease])
    Gap_Am_mean_list.append(dfAm1[disease])
    CIAm_list.append(dfAm2[disease])
    Mean_list.append(dfAm1[disease])
    # print("----------------------")
    # print(Mean_list)

    cleanedList_mean = [x for x in Mean_list if str(x) != 'nan']
    # print(cleanedList_mean)
    Distance_list.append(np.max(cleanedList_mean) - np.min(cleanedList_mean))
    # print(Distance_list)
    diseases_abbr_list.append(diseases_abbr[disease])

d = {'diseases': diseases, 'diseases_abbr': diseases_abbr_list, 'Distance': Distance_list,
     "%White": PrcentW_list, 'Gap_W_mean': Gap_W_mean_list, 'CI_W': CIW_list,
     "%Black": PrcentB_list, 'Gap_B_mean': Gap_B_mean_list, 'CI_B': CIB_list,
     "%Hisp": PrcentH_list, 'Gap_H_mean': Gap_H_mean_list, 'CI_H': CIH_list,
     "%Other": PrcentOt_list, 'Gap_Ot_mean': Gap_Ot_mean_list, 'CI_Ot': CIOt_list,
     "%Asian": PrcentAs_list, 'Gap_As_mean': Gap_As_mean_list, 'CI_As': CIAs_list,
     "%American": PrcentAm_list, 'Gap_Am_mean': Gap_Am_mean_list, 'CI_Am': CIAm_list
     }
WantR = pd.DataFrame(d)


WantR = WantR.sort_values(by ='Distance' )
WantR.to_csv("./results/WantR.csv")

plt.rcParams.update({'font.size': F})

plt.figure(figsize=(16,m))

plt.errorbar(WantR['diseases_abbr'],WantR['Gap_W_mean'],yerr = WantR['CI_W'],fmt='o',mfc='blue')#ecolor='blue'
plt.scatter(WantR['diseases_abbr'],WantR['Gap_W_mean'],s= np.multiply(WantR['%White'],500), marker='o',color='blue', label="WHITE")

plt.scatter(WantR['diseases_abbr'],WantR['Gap_B_mean'],s= np.multiply(WantR['%Black'],500), marker='o',color='orange', label="BLACK")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_B_mean'],yerr = WantR['CI_B'],fmt='o',mfc='orange')

plt.scatter(WantR['diseases_abbr'],WantR['Gap_H_mean'],s= np.multiply(WantR['%Hisp'],500), marker='o',color='green', label="HISPANIC")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_H_mean'],yerr = WantR['CI_H'],fmt='o',mfc='green')

plt.scatter(WantR['diseases_abbr'],WantR['Gap_Ot_mean'],s= np.multiply(WantR['%Other'],500), marker='o',color='r', label="OTHER")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_Ot_mean'],yerr = WantR['CI_Ot'],fmt='o',mfc='r')

plt.scatter(WantR['diseases_abbr'],WantR['Gap_As_mean'],s= np.multiply(WantR['%Asian'],500), marker='o',color='m', label="ASIAN")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_As_mean'],yerr = WantR['CI_As'],fmt='o',mfc='m')

plt.scatter(WantR['diseases_abbr'],WantR['Gap_Am_mean'],s= np.multiply(WantR['%American'],500), marker='o',color='k', label="NATIVE")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_Am_mean'],yerr = WantR['CI_Am'],fmt='o',mfc='k')


plt.ylabel("TPR RACE DISPARITY")
plt.legend()
plt.grid(True)
plt.savefig("./results/RACE3.pdf")



plt.rcParams.update({'font.size': F})

plt.figure(figsize=(16,m))

plt.errorbar(WantR['diseases_abbr'],WantR['Gap_W_mean'],yerr = WantR['CI_W'],fmt='o',mfc='blue')#ecolor='blue'
plt.scatter(WantR['diseases_abbr'],WantR['Gap_W_mean'],s= np.multiply(WantR['%White'],500), marker='o',color='blue', label="WHITE")

plt.scatter(WantR['diseases_abbr'],WantR['Gap_B_mean'],s= np.multiply(WantR['%Black'],500), marker='o',color='orange', label="BLACK")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_B_mean'],yerr = WantR['CI_B'],fmt='o',mfc='orange')

plt.scatter(WantR['diseases_abbr'],WantR['Gap_H_mean'],s= np.multiply(WantR['%Hisp'],500), marker='o',color='green', label="HISPANIC")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_H_mean'],yerr = WantR['CI_H'],fmt='o',mfc='green')

plt.scatter(WantR['diseases_abbr'],WantR['Gap_Ot_mean'],s= np.multiply(WantR['%Other'],500), marker='o',color='r', label="OTHER")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_Ot_mean'],yerr = WantR['CI_Ot'],fmt='o',mfc='r')

plt.scatter(WantR['diseases_abbr'],WantR['Gap_As_mean'],s= np.multiply(WantR['%Asian'],500), marker='o',color='m', label="ASIAN")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_As_mean'],yerr = WantR['CI_As'],fmt='o',mfc='m')

plt.scatter(WantR['diseases_abbr'],WantR['Gap_Am_mean'],s= np.multiply(WantR['%American'],500), marker='o',color='k', label="NATIVE")
plt.errorbar(WantR['diseases_abbr'],WantR['Gap_Am_mean'],yerr = WantR['CI_Am'],fmt='o',mfc='k')

plt.xticks(rotation=deg)
plt.ylabel("TPR RACE DISPARITY")
plt.legend()
plt.grid(True)
plt.savefig("./results/TPR_Dis_RACE.pdf")

# fig, ax = plt.subplots()
# ax.scatter(WantR['diseases_abbr'],WantR['Gap_W_mean'],s= np.multiply(WantR['%White'],500), marker='o',color='blue', label="WHITE")
# ax.scatter(WantR['diseases_abbr'],WantR['Gap_B_mean'],s= np.multiply(WantR['%Black'],500), marker='o',color='orange', label="BLACK")
# ax.scatter(WantR['diseases_abbr'],WantR['Gap_Ot_mean'],s= np.multiply(WantR['%Other'],500), marker='o',color='pink', label="OTHER")
# ax.scatter(WantR['diseases_abbr'],WantR['Gap_Am_mean'],s= np.multiply(WantR['%American'],500), marker='o',color='purple', label="NATIVE")
# ax.scatter(WantR['diseases_abbr'],WantR['Gap_As_mean'],s= np.multiply(WantR['%Asian'],500), marker='o',color='red', label="Asian")
# ax.legend()
# plt.savefig("./results/RACE2.pdf")
#---------------------------------------------       Insurance

resultI = Run1_insurance.append([Run2_insurance, Run3_insurance,Run4_insurance, Run5_insurance])

#resultI.to_csv("./results/resultI.csv")

resI =resultI.groupby("diseases")
resI_df = resI.describe()

#resI_df.to_csv("./results/resI.csv")

#print(resI.columns)

WantI = pd.DataFrame(pd.DataFrame(diseases, columns=["diseases"]))

dfC0 = resI_df['%Medicare']['mean']
dfC1 = resI_df['Gap_Medicare']["mean"]
dfC2 = 1.96 * resI_df['Gap_Medicare']["std"] / np.sqrt(5)


dfO0 = resI_df['%Other']['mean']
dfO1 = resI_df['Gap_Other']["mean"]
dfO2 = 1.96 * resI_df['Gap_Other']["std"] / np.sqrt(5)

dfA0 = resI_df['%Medicaid']['mean']
dfA1 = resI_df['Gap_Medicaid']["mean"]
dfA2 = 1.96 * resI_df['Gap_Medicaid']["std"] / np.sqrt(5)



PrcentA_list = []
CIA_list = []
Gap_A_mean_list = []



PrcentC_list = []
CIC_list = []
Gap_C_mean_list = []

PrcentO_list = []
CIO_list = []
Gap_O_mean_list = []

diseases_abbr_list = []
Distance_list = []


for disease in diseases:
    Mean_list = []

    PrcentO_list.append(dfO0[disease])
    Gap_O_mean_list.append(dfO1[disease])
    CIO_list.append(dfO2[disease])
    Mean_list.append(dfO1[disease])


    PrcentC_list.append(dfC0[disease])
    Gap_C_mean_list.append(dfC1[disease])
    CIC_list.append(dfC2[disease])
    Mean_list.append(dfC1[disease])

    PrcentA_list.append(dfA0[disease])
    Gap_A_mean_list.append(dfA1[disease])
    CIA_list.append(dfA2[disease])
    Mean_list.append(dfA1[disease])


    cleanedList_mean = [x for x in Mean_list if str(x) != 'nan']

    Distance_list.append(np.max(cleanedList_mean) - np.min(cleanedList_mean))
    diseases_abbr_list.append(diseases_abbr[disease])

d = {'diseases': diseases, 'diseases_abbr': diseases_abbr_list, 'Distance': Distance_list,
     "%Medicare": PrcentC_list, 'Gap_C_mean': Gap_C_mean_list, 'CI_C': CIC_list,
     "%Other": PrcentO_list, 'Gap_O_mean': Gap_O_mean_list, 'CI_O': CIO_list,
     "%Medicaid": PrcentA_list, 'Gap_A_mean': Gap_A_mean_list, 'CI_A': CIA_list

     }
WantI = pd.DataFrame(d)


WantI = WantI.sort_values(by ='Distance' )
WantI.to_csv("./results/WantI.csv")

WantX.to_csv("./results/WantX.csv")

plt.rcParams.update({'font.size': F})

plt.figure(figsize=(16,m))
plt.scatter(WantI['diseases_abbr'],WantI['Gap_C_mean'],s= np.multiply(WantI['%Medicare'],500), marker='o',color='blue', label="MEDICARE")
plt.errorbar(WantI['diseases_abbr'],WantI['Gap_C_mean'],yerr = WantI['CI_C'],fmt='o',mfc='blue')

plt.scatter(WantI['diseases_abbr'],WantI['Gap_O_mean'],s= np.multiply(WantI['%Other'],500), marker='o',color='orange', label="OTHER")
plt.errorbar(WantI['diseases_abbr'],WantI['Gap_O_mean'],yerr = WantI['CI_O'],fmt='o',mfc='orange')

plt.scatter(WantI['diseases_abbr'],WantI['Gap_A_mean'],s= np.multiply(WantI['%Medicaid'],500), marker='o',color='green', label="MEDICAID")
plt.errorbar(WantI['diseases_abbr'],WantI['Gap_A_mean'],yerr = WantI['CI_A'],fmt='o',mfc='green')


plt.xticks(rotation=deg)
plt.ylabel("TPR INSURANCE DISPARITY")
plt.legend()
plt.grid(True)
plt.savefig("./results/TPR_Dis_INSURANCE.pdf")

# #------------------------------------------- Percentages

# train_df_path ="/h/laleh/Desktop/PycharmProjects/Fairness/Nov7/Percentage/MIMIC/train.csv"
# test_df_path ="/h/laleh/Desktop/PycharmProjects/Fairness/Nov7/Percentage/MIMIC/test.csv"
# val_df_path = "/h/laleh/Desktop/PycharmProjects/Fairness/Nov7/Percentage/MIMIC/valid.csv"
# val_df = pd.read_csv(val_df_path)
# train_df = pd.read_csv(train_df_path)
# test_df = pd.read_csv(test_df_path)
# WholeData = test_df.append([val_df, train_df])

# #------------ Sex
# WholeDataX =WholeData.groupby("gender_x")
# WholeDataX_df = WholeDataX.describe()
# dfWhole_Sex = WholeDataX_df["subject_id"]['count']
# total_CXP= dfWhole_Sex["F"]+dfWhole_Sex["M"]
# Male_percent = 100*dfWhole_Sex["M"]/total_CXP
# print("Male Percent:  ",Male_percent)

# print("female Percent:  ",100*dfWhole_Sex["F"]/total_CXP)
# print("#images",total_CXP)

# #----------  Age
# WholeDataA =WholeData.groupby("age_decile")
# WholeDataA_df = WholeDataA.describe()
# total_CXR = WholeDataA_df["subject_id"]['count'].sum()
# dfWhole_Age = WholeDataA_df["subject_id"]['count']
# Age1 = 100*(dfWhole_Age["0-10"]+dfWhole_Age["20-Oct"])/total_CXR
# Age2 = 100*(dfWhole_Age["20-30"]+dfWhole_Age["30-40"])/total_CXR
# Age3 = 100*(dfWhole_Age["40-50"]+dfWhole_Age["50-60"])/total_CXR
# Age4 = 100*(dfWhole_Age["60-70"]+dfWhole_Age["70-80"])/total_CXR
# Age5 = 100*(dfWhole_Age["80-90"]+dfWhole_Age[">=90"])/total_CXR
# print("0-20 Percent:  ",Age1)
# print("20-40 Percent:  ",Age2)
# print("40-60 Percent:  ",Age3)
# print("60-80 Percent:  ",Age4)
# print("80- Percent:  ",Age5)

# WholeDataI =WholeData.groupby("insurance")
# WholeDataI_df = WholeDataI.describe()
# total_CXR = WholeDataI_df["subject_id"]['count'].sum()
# dfWhole_Ins = WholeDataI_df["subject_id"]['count']
# Ins1 = 100*(dfWhole_Ins["Medicare"])/total_CXR
# Ins2 = 100*(dfWhole_Ins["Medicaid"])/total_CXR
# Ins3 = 100*(dfWhole_Ins["Other"])/total_CXR

# print("Medicare Percent:  ",Ins1)
# print("Medicaid Percent:  ",Ins2)
# print("Other Percent:  ",Ins3)


# #---- Race
# WholeDataR =WholeData.groupby("race")
# WholeDataR_df = WholeDataR.describe()

# WholeDataR =WholeData.groupby("race")
# WholeDataR_df = WholeDataR.describe()
# total_CXR = WholeDataR_df["subject_id"]['count'].sum()
# dfWhole_Race = WholeDataR_df["subject_id"]['count']
# Rac1 = 100*(dfWhole_Race["WHITE"])/total_CXR
# Rac2 = 100*(dfWhole_Race["BLACK/AFRICAN AMERICAN"])/total_CXR
# Rac3 = 100*(dfWhole_Race["OTHER"])/total_CXR
# Rac4 = 100*(dfWhole_Race["ASIAN"])/total_CXR
# Rac5 = 100*(dfWhole_Race["HISPANIC/LATINO"])/total_CXR
# Rac6 = 100*(dfWhole_Race["AMERICAN INDIAN/ALASKA NATIVE"])/total_CXR

# print("WHITE Percent:  ",Rac1)
# print("BLACK/AFRICAN AMERICAN Percent:  ",Rac2)
# print("OTHER Percent:  ",Rac3)
# print("ASIAN Percent:  ",Rac4)
# print("HISPANIC/LATINO Percent:  ",Rac5)
# print("AMERICAN INDIAN/ALASKA NATIVE Percent:  ",Rac6)
# print("Unkhonwn or unable to obtain Percent:  ",100- (Rac1 + Rac2 + Rac3 +Rac4 + Rac5 + Rac6))

#-               FPR confidence intervals

# FP5_age = pd.read_csv("./results22/FP5_age.csv")
# FP4_age = pd.read_csv("./results19/FP4_age.csv")
# FP3_age = pd.read_csv("./results38/FP3_age.csv")
# FP2_age = pd.read_csv("./results47/FP2_age.csv")
# FP1_age = pd.read_csv("./results31/FP1_age.csv")
# FP_age = FP1_age.append([FP2_age, FP3_age,FP4_age, FP5_age])
# F_age_df = FP_age.describe()

# F_age_df.loc['mean']
# 1.96 * F_age_df.loc['std'] / np.sqrt(5)

# FP5_race = pd.read_csv("./results22/FP5_race.csv")
# FP4_race = pd.read_csv("./results19/FP4_race.csv")
# FP3_race = pd.read_csv("./results38/FP3_race.csv")
# FP2_race = pd.read_csv("./results47/FP2_race.csv")
# FP1_race = pd.read_csv("./results31/FP1_race.csv")
# FP_race = FP1_race.append([FP2_race, FP3_race,FP4_race, FP5_race])
# F_race_df = FP_race.describe()

# F_race_df.loc['mean']
# 1.96 * F_race_df.loc['std'] / np.sqrt(5)

# FP5_insurance = pd.read_csv("./results22/FP5_insurance.csv")
# FP4_insurance = pd.read_csv("./results19/FP4_insurance.csv")
# FP3_insurance = pd.read_csv("./results38/FP3_insurance.csv")
# FP2_insurance = pd.read_csv("./results47/FP2_insurance.csv")
# FP1_insurance = pd.read_csv("./results31/FP1_insurance.csv")
# FP_insurance = FP1_insurance.append([FP2_insurance, FP3_insurance,FP4_insurance, FP5_insurance])
# FP_insurance_df = FP_insurance.describe()

# FP_insurance_df.loc['mean']
# 1.96 * FP_insurance_df.loc['std'] / np.sqrt(5)

# fig, ax = plt.subplots(figsize=(18,4))

# ages = [ '80-', '60-80', '20-40', '40-60', '0-20']

# ages_pos = np.arange(len(ages))

# # This values are calculated in Confidence.ipynb
# CTEs = [0.084,  0.116, 0.264,0.271,0.445]
# error = [0.008, 0.006, 0.010,0.008,0.028]

# ax.bar(ages_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)

# races = [ 'WHITE', 'NATIVE', 'OTHER', 'ASIAN','HIPANIC', 'BLACK']
# race_pos = np.arange(5,len(races)+5)

# CTEs = [0.163,  0.166, 0.195,0.213,0.233,0.249]
# error = [0.007, 0.023, 0.011,0.011,0.011,0.007]

# ax.bar(race_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)


# Insurances = [ 'MEDICARE', 'OTHER', 'MEDICAID']

# Insurances_pos = np.arange(11,len(Insurances)+11)

# CTEs = [0.163,  0.179, 0.307]
# error = [0.006, 0.005, 0.015]

# ax.bar(Insurances_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)

# labels = ['80-', '60-80', '20-40', '40-60', '0-20','WHITE', 'NATIVE', 'OTHER', 'ASIAN','HISPANIC', 'BLACK', 'MEDICARE', 'OTHER', 'MEDICAID']
# x_pos = np.arange(len(labels))
# y_labels = ['0.0', '0.1', '0.2', '0.3', '0.4']
# #ax.set_ylabel('FALSE POSITIVE RATE')
# ax.set_ylabel('"NF" FALSE POSITIVE RATE',fontsize = 14.0)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(labels,fontsize = 14.0)
# ax.set_yticklabels(y_labels,fontsize = 14.0)
# #ax.set_title('A) AGE                                                           B)RACE                                                       C)INSURANCE')
# ax.text(1,0.45,'A) AGE                                                                      B)RACE                                                  C)INSURANCE',fontsize = 14.0)
# ax.yaxis.grid(True)
# fig.savefig('./results/FP.pdf')

# #-------------------------------------   % 5 run AUC
