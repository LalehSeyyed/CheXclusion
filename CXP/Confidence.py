import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Run5_sex = pd.read_csv("./results90/Run5_sex.csv")
Run4_sex = pd.read_csv("./results56/Run4_sex.csv")
Run3_sex = pd.read_csv("./results60/Run3_sex.csv")
Run2_sex = pd.read_csv("./results32/Run2_sex.csv")
Run1_sex = pd.read_csv("./results40/Run1_sex.csv")

diseases = ['Atelectasis',  'Cardiomegaly', 'Consolidation',
            'Edema' ,  'Enlarged Cardiomediastinum','Fracture',
            'Lung Lesion','Lung Opacity',  'No Finding', 'Pleural Effusion','Pleural Other',"Pneumonia",
            'Pneumothorax', 'Support Devices' ]
diseases_abbr = {'Lung Opacity': 'AO',
                'Cardiomegaly': 'Cd',
                'Effusion': 'Ef',
                'Enlarged Cardiomediastinum': 'EC',
                'Lung Lesion': 'LL',
                'Atelectasis': 'A',
                'Pneumonia': 'Pa',
                'Pneumothorax': 'Px',
                'Consolidation': 'Co',
                'Edema': 'Ed',
                'Pleural Effusion': 'Ef',
                'Pleural Other': 'PO',
                'Fracture': 'Fr',
                'Support Devices': 'SD',
                'Airspace Opacity': 'AO',
                'No Finding': 'NF'
                }


diseases_abbr = {'Lung Opacity':'Lung Opacity',
                'Cardiomegaly': 'Cardiomegaly',
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
WantX.to_csv("./results/WantX.csv")

plt.rcParams.update({'font.size': F})

plt.figure(figsize=(16,m))
plt.scatter(WantX['diseases_abbr'],WantX['Gap_M_mean'],s= np.multiply(WantX['%M'],500), marker='o',color='blue', label="Male")
plt.errorbar(WantX['diseases_abbr'],WantX['Gap_M_mean'],yerr = WantX['CI_M'],fmt='o',mfc='blue')
plt.scatter(WantX['diseases_abbr'],WantX['Gap_F_mean'],s= np.multiply(WantX['%F'],500), marker='o',color='red', label="Female")
plt.errorbar(WantX['diseases_abbr'],WantX['Gap_F_mean'],yerr = WantX['CI_F'],fmt='o',mfc='red')

plt.xticks(rotation = deg)
plt.ylabel("TPR SEX DISPARITY")
plt.legend()
plt.grid(True)
plt.savefig("./results/SEX_TPR_DIS_CXP.pdf")

#---------------------------------------------------   Age
Run5_Age = pd.read_csv("./results90/Run5_Age.csv")
Run4_Age = pd.read_csv("./results56/Run4_Age.csv")
Run3_Age = pd.read_csv("./results60/Run3_Age.csv")
Run2_Age = pd.read_csv("./results32/Run2_Age.csv")
Run1_Age = pd.read_csv("./results40/Run1_Age.csv")


resultAge = Run1_Age.append([Run2_Age, Run3_Age, Run4_Age, Run5_Age])
resAge =resultAge.groupby("diseases")
resAge_df = resAge.describe()

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
plt.savefig("./results/AGE_TPR_DIS_CXP.pdf")
