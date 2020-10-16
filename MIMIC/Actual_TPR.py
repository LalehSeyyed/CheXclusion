import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit






def TPR_14(TrueWithMeta_df, df, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
    df = df.merge(TrueWithMeta_df, left_on="path", right_on="path")

    GAP_total = []
    percentage_total = []
    Total_total = []
    Positive_total = []
    Negetive_total = []    
    
    
    cate = []

    print(diseases)

    if category_name == 'gender':
        Run1_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile':
        Run1_age = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'race':
        Run1_race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'insurance':
        Run1_insurance = pd.DataFrame(diseases, columns=["diseases"])

    for c in category:
        GAP_y = []
        percentage_y = []
        Total_y = []
        Positive_y = []
        Negetive_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            
        #within category (e.g Male) with positive for disease d    
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :] 
        
        #within category (e.g Male) with negative for disease d
            Ne_gy = df.loc[(df[d] == 0) & (df[category_name] == c), :] 
        
        # All subgroups wihin a category that have disease (e.g male and female with disease d)
        #df[category_name] != 0 means all becouse we need to not to consider an image if we do not have the its meta-data
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]
            

            Total = len(pi_gy) + len(Ne_gy)
            Positive = len(pi_gy)
            Negetive = len(Ne_gy)
                        
            
            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                percentage = len(pi_gy) / len(pi_y)
                GAP = TPR  # just to not to update parameter name later
                GAP_y.append(GAP)
                percentage_y.append(percentage)
                
            else:
                GAP_y.append(np.NaN)
                percentage_y.append(0)
                
            
            Total_y.append(Total)
            Positive_y.append(Positive)
            Negetive_y.append(Negetive) 

        # Gaps of all 14 diseases and categories
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        Total_total.append(Total_y)
        Positive_total.append(Positive_y)
        Negetive_total.append(Negetive_y) 




    for i in range(len(GAP_total)):




        if category_name == 'age_decile':

            if i == 0:
                Percent4 = pd.DataFrame(percentage_total[i], columns=["%60-80"])
                Run1_age = pd.concat([Run1_age, Percent4.reindex(Run1_age.index)], axis=1)
                
                posit4 = pd.DataFrame(Positive_total[i], columns=["P60-80"])
                Run1_age = pd.concat([Run1_age, posit4.reindex(Run1_age.index)], axis=1)
                
                Negat4 = pd.DataFrame(Negetive_total[i], columns=["N60-80"])
                Run1_age = pd.concat([Run1_age, Negat4.reindex(Run1_age.index)], axis=1)
                
                Total4 = pd.DataFrame(Total_total[i], columns=["Tot60-80"])
                Run1_age = pd.concat([Run1_age, Total4.reindex(Run1_age.index)], axis=1)

                Gap4 = pd.DataFrame(GAP_total[i], columns=["TPR_60-80"])
                Run1_age = pd.concat([Run1_age, Gap4.reindex(Run1_age.index)], axis=1)

            if i == 1:
                Percent6 = pd.DataFrame(percentage_total[i], columns=["%40-60"])
                Run1_age = pd.concat([Run1_age, Percent6.reindex(Run1_age.index)], axis=1)
                
                posit6 = pd.DataFrame(Positive_total[i], columns=["P40-60"])
                Run1_age = pd.concat([Run1_age, posit6.reindex(Run1_age.index)], axis=1)
                
                Negat6 = pd.DataFrame(Negetive_total[i], columns=["N40-60"])
                Run1_age = pd.concat([Run1_age, Negat6.reindex(Run1_age.index)], axis=1)
                
                Total6 = pd.DataFrame(Total_total[i], columns=["Tot40-60"])
                Run1_age = pd.concat([Run1_age, Total6.reindex(Run1_age.index)], axis=1)

                Gap6 = pd.DataFrame(GAP_total[i], columns=["TPR_40-60"])
                Run1_age = pd.concat([Run1_age, Gap6.reindex(Run1_age.index)], axis=1)

            if i == 2:
                Percent2 = pd.DataFrame(percentage_total[i], columns=["%20-40"])
                Run1_age = pd.concat([Run1_age, Percent2.reindex(Run1_age.index)], axis=1)
                
                posit2 = pd.DataFrame(Positive_total[i], columns=["P20-40"])
                Run1_age = pd.concat([Run1_age, posit2.reindex(Run1_age.index)], axis=1)
                
                Negat2 = pd.DataFrame(Negetive_total[i], columns=["N20-40"])
                Run1_age = pd.concat([Run1_age, Negat2.reindex(Run1_age.index)], axis=1)
                
                Total2 = pd.DataFrame(Total_total[i], columns=["Tot20-40"])
                Run1_age = pd.concat([Run1_age, Total2.reindex(Run1_age.index)], axis=1)
                
                Gap2 = pd.DataFrame(GAP_total[i], columns=["TPR_20-40"])
                Run1_age = pd.concat([Run1_age, Gap2.reindex(Run1_age.index)], axis=1)

            if i == 3:
                Percent8 = pd.DataFrame(percentage_total[i], columns=["%80-"])
                Run1_age = pd.concat([Run1_age, Percent8.reindex(Run1_age.index)], axis=1)
                
                posit8 = pd.DataFrame(Positive_total[i], columns=["P80-"])
                Run1_age = pd.concat([Run1_age, posit8.reindex(Run1_age.index)], axis=1)
                
                Negat8 = pd.DataFrame(Negetive_total[i], columns=["N80-"])
                Run1_age = pd.concat([Run1_age, Negat8.reindex(Run1_age.index)], axis=1)
                
                Total8 = pd.DataFrame(Total_total[i], columns=["Tot80-"])
                Run1_age = pd.concat([Run1_age, Total8.reindex(Run1_age.index)], axis=1)
                
                Gap8 = pd.DataFrame(GAP_total[i], columns=["TPR_80-"])
                Run1_age = pd.concat([Run1_age, Gap8.reindex(Run1_age.index)], axis=1)

            if i == 4:
                Percent0 = pd.DataFrame(percentage_total[i], columns=["%0-20"])
                Run1_age = pd.concat([Run1_age, Percent0.reindex(Run1_age.index)], axis=1)

                posit0 = pd.DataFrame(Positive_total[i], columns=["P0-20"])
                Run1_age = pd.concat([Run1_age, posit0.reindex(Run1_age.index)], axis=1)
                
                Negat0 = pd.DataFrame(Negetive_total[i], columns=["N0-20"])
                Run1_age = pd.concat([Run1_age, Negat0.reindex(Run1_age.index)], axis=1)
                
                Total0 = pd.DataFrame(Total_total[i], columns=["Tot0-20"])
                Run1_age = pd.concat([Run1_age, Total0.reindex(Run1_age.index)], axis=1)                
                
                Gap0 = pd.DataFrame(GAP_total[i], columns=["TPR_0-20"])
                Run1_age = pd.concat([Run1_age, Gap0.reindex(Run1_age.index)], axis=1)

            Run1_age.to_csv("./results/Run1_TPR_Age.csv")

        if category_name == 'gender':

            if i == 0:
                MalePercent = pd.DataFrame(percentage_total[i], columns=["%M"])
                Run1_sex = pd.concat([Run1_sex, MalePercent.reindex(Run1_sex.index)], axis=1)
                
                positMale = pd.DataFrame(Positive_total[i], columns=["P_M"])
                Run1_sex = pd.concat([Run1_sex, positMale.reindex(Run1_sex.index)], axis=1)
                
                NegatMale = pd.DataFrame(Negetive_total[i], columns=["N_M"])
                Run1_sex = pd.concat([Run1_sex, NegatMale.reindex(Run1_sex.index)], axis=1)
                
                TotalMale = pd.DataFrame(Total_total[i], columns=["Tot_M"])
                Run1_sex = pd.concat([Run1_sex, TotalMale.reindex(Run1_sex.index)], axis=1)                 

                MaleGap = pd.DataFrame(GAP_total[i], columns=["TPR_M"])
                Run1_sex = pd.concat([Run1_sex, MaleGap.reindex(Run1_sex.index)], axis=1)

            else:
                FeMalePercent = pd.DataFrame(percentage_total[i], columns=["%F"])
                Run1_sex = pd.concat([Run1_sex, FeMalePercent.reindex(Run1_sex.index)], axis=1)
                
                positFeMale = pd.DataFrame(Positive_total[i], columns=["P_F"])
                Run1_sex = pd.concat([Run1_sex, positFeMale.reindex(Run1_sex.index)], axis=1)
                
                NegatFeMale = pd.DataFrame(Negetive_total[i], columns=["N_F"])
                Run1_sex = pd.concat([Run1_sex, NegatFeMale.reindex(Run1_sex.index)], axis=1)
                
                TotalFeMale = pd.DataFrame(Total_total[i], columns=["Tot_F"])
                Run1_sex = pd.concat([Run1_sex, TotalFeMale.reindex(Run1_sex.index)], axis=1)                  

                FeMaleGap = pd.DataFrame(GAP_total[i], columns=["TPR_F"])
                Run1_sex = pd.concat([Run1_sex, FeMaleGap.reindex(Run1_sex.index)], axis=1)

            Run1_sex.to_csv("./results/Run1_TPR_sex.csv")

        if category_name == 'race':
            if i == 0:
                WhPercent = pd.DataFrame(percentage_total[i], columns=["%White"])
                Run1_race = pd.concat([Run1_race, WhPercent.reindex(Run1_race.index)], axis=1)
                
                positWh = pd.DataFrame(Positive_total[i], columns=["P_Wh"])
                Run1_race = pd.concat([Run1_race, positWh.reindex(Run1_race.index)], axis=1)
                
                NegatWh = pd.DataFrame(Negetive_total[i], columns=["N_Wh"])
                Run1_race = pd.concat([Run1_race, NegatWh.reindex(Run1_race.index)], axis=1)
                
                TotalWh = pd.DataFrame(Total_total[i], columns=["Tot_Wh"])
                Run1_race = pd.concat([Run1_race, TotalWh.reindex(Run1_race.index)], axis=1)                   
                

                WhGap = pd.DataFrame(GAP_total[i], columns=["TPR_White"])
                Run1_race = pd.concat([Run1_race, WhGap.reindex(Run1_race.index)], axis=1)

            if i == 1:
                BlPercent = pd.DataFrame(percentage_total[i], columns=["%Black"])
                Run1_race = pd.concat([Run1_race, BlPercent.reindex(Run1_race.index)], axis=1)
                
                positBl = pd.DataFrame(Positive_total[i], columns=["P_Bl"])
                Run1_race = pd.concat([Run1_race, positBl.reindex(Run1_race.index)], axis=1)
                
                NegatBl = pd.DataFrame(Negetive_total[i], columns=["N_Bl"])
                Run1_race = pd.concat([Run1_race, NegatBl.reindex(Run1_race.index)], axis=1)
                
                TotalBl = pd.DataFrame(Total_total[i], columns=["Tot_Bl"])
                Run1_race = pd.concat([Run1_race, TotalBl.reindex(Run1_race.index)], axis=1) 
                
                BlGap = pd.DataFrame(GAP_total[i], columns=["TPR_Black"])
                Run1_race = pd.concat([Run1_race, BlGap.reindex(Run1_race.index)], axis=1)

            if i == 2:
                BlPercent = pd.DataFrame(percentage_total[i], columns=["%Hisp"])
                Run1_race = pd.concat([Run1_race, BlPercent.reindex(Run1_race.index)], axis=1)

                positHi = pd.DataFrame(Positive_total[i], columns=["P_Hi"])
                Run1_race = pd.concat([Run1_race, positHi.reindex(Run1_race.index)], axis=1)
                
                NegatHi = pd.DataFrame(Negetive_total[i], columns=["N_Hi"])
                Run1_race = pd.concat([Run1_race, NegatHi.reindex(Run1_race.index)], axis=1)
                
                TotalHi = pd.DataFrame(Total_total[i], columns=["Tot_Hi"])
                Run1_race = pd.concat([Run1_race, TotalHi.reindex(Run1_race.index)], axis=1)                 
                
                BlGap = pd.DataFrame(GAP_total[i], columns=["TPR_Hisp"])
                Run1_race = pd.concat([Run1_race, BlGap.reindex(Run1_race.index)], axis=1)

            if i == 3:
                OtPercent = pd.DataFrame(percentage_total[i], columns=["%Other"])
                Run1_race = pd.concat([Run1_race, OtPercent.reindex(Run1_race.index)], axis=1)

                positOt = pd.DataFrame(Positive_total[i], columns=["P_Ot"])
                Run1_race = pd.concat([Run1_race, positOt.reindex(Run1_race.index)], axis=1)
                
                NegatOt = pd.DataFrame(Negetive_total[i], columns=["N_Ot"])
                Run1_race = pd.concat([Run1_race, NegatOt.reindex(Run1_race.index)], axis=1)
                
                TotalOt = pd.DataFrame(Total_total[i], columns=["Tot_Ot"])
                Run1_race = pd.concat([Run1_race, TotalOt.reindex(Run1_race.index)], axis=1)   
                
                OtGap = pd.DataFrame(GAP_total[i], columns=["TPR_Other"])
                Run1_race = pd.concat([Run1_race, OtGap.reindex(Run1_race.index)], axis=1)

            if i == 4:
                AsPercent = pd.DataFrame(percentage_total[i], columns=["%Asian"])
                Run1_race = pd.concat([Run1_race, AsPercent.reindex(Run1_race.index)], axis=1)

                positAs = pd.DataFrame(Positive_total[i], columns=["P_As"])
                Run1_race = pd.concat([Run1_race, positAs.reindex(Run1_race.index)], axis=1)
                
                NegatAs = pd.DataFrame(Negetive_total[i], columns=["N_As"])
                Run1_race = pd.concat([Run1_race, NegatAs.reindex(Run1_race.index)], axis=1)
                
                TotalAs = pd.DataFrame(Total_total[i], columns=["Tot_As"])
                Run1_race = pd.concat([Run1_race, TotalAs.reindex(Run1_race.index)], axis=1)   
                
                AsGap = pd.DataFrame(GAP_total[i], columns=["Gap_Asian"])
                Run1_race = pd.concat([Run1_race, AsGap.reindex(Run1_race.index)], axis=1)

            if i == 5:
                AmPercent = pd.DataFrame(percentage_total[i], columns=["%American"])
                Run1_race = pd.concat([Run1_race, AmPercent.reindex(Run1_race.index)], axis=1)
                
                positAm = pd.DataFrame(Positive_total[i], columns=["P_Am"])
                Run1_race = pd.concat([Run1_race, positAm.reindex(Run1_race.index)], axis=1)
                
                NegatAm = pd.DataFrame(Negetive_total[i], columns=["N_Am"])
                Run1_race = pd.concat([Run1_race, NegatAm.reindex(Run1_race.index)], axis=1)
                
                TotalAm = pd.DataFrame(Total_total[i], columns=["Tot_Am"])
                Run1_race = pd.concat([Run1_race, TotalAm.reindex(Run1_race.index)], axis=1)                   
                
                AmGap = pd.DataFrame(GAP_total[i], columns=["TPR_American"])
                Run1_race = pd.concat([Run1_race, AmGap.reindex(Run1_race.index)], axis=1)

            Run1_race.to_csv("./results/Run1_TPR_race.csv")

        if category_name == 'insurance':
            if i == 0:
                CarePercent = pd.DataFrame(percentage_total[i], columns=["%Medicare"])
                Run1_insurance = pd.concat([Run1_insurance, CarePercent.reindex(Run1_insurance.index)], axis=1)

                positCare = pd.DataFrame(Positive_total[i], columns=["P_Care"])
                Run1_insurance = pd.concat([Run1_insurance, positCare.reindex(Run1_insurance.index)], axis=1)
                
                NegatCare = pd.DataFrame(Negetive_total[i], columns=["N_Care"])
                Run1_insurance = pd.concat([Run1_insurance, NegatCare.reindex(Run1_insurance.index)], axis=1)
                
                TotalCare = pd.DataFrame(Total_total[i], columns=["Tot_Care"])
                Run1_insurance = pd.concat([Run1_insurance, TotalCare.reindex(Run1_insurance.index)], axis=1)                  
                
                CareGap = pd.DataFrame(GAP_total[i], columns=["TPR_Medicare"])
                Run1_insurance = pd.concat([Run1_insurance, CareGap.reindex(Run1_insurance.index)], axis=1)

            if i == 1:
                OtherPercent = pd.DataFrame(percentage_total[i], columns=["%Other"])
                Run1_insurance = pd.concat([Run1_insurance, OtherPercent.reindex(Run1_insurance.index)], axis=1)

                positOt = pd.DataFrame(Positive_total[i], columns=["P_Ot"])
                Run1_insurance = pd.concat([Run1_insurance, positOt.reindex(Run1_insurance.index)], axis=1)
                
                NegatOt = pd.DataFrame(Negetive_total[i], columns=["N_Ot"])
                Run1_insurance = pd.concat([Run1_insurance, NegatOt.reindex(Run1_insurance.index)], axis=1)
                
                TotalOt = pd.DataFrame(Total_total[i], columns=["Tot_Ot"])
                Run1_insurance = pd.concat([Run1_insurance, TotalOt.reindex(Run1_insurance.index)], axis=1)                  
                
                OtherGap = pd.DataFrame(GAP_total[i], columns=["TPR_Other"])
                Run1_insurance = pd.concat([Run1_insurance, OtherGap.reindex(Run1_insurance.index)], axis=1)

            if i == 2:
                AidPercent = pd.DataFrame(percentage_total[i], columns=["%Medicaid"])
                Run1_insurance = pd.concat([Run1_insurance, AidPercent.reindex(Run1_insurance.index)], axis=1)
                
                positAid = pd.DataFrame(Positive_total[i], columns=["P_Aid"])
                Run1_insurance = pd.concat([Run1_insurance, positAid.reindex(Run1_insurance.index)], axis=1)
                
                NegatAid = pd.DataFrame(Negetive_total[i], columns=["N_Aid"])
                Run1_insurance = pd.concat([Run1_insurance, NegatAid.reindex(Run1_insurance.index)], axis=1)
                
                TotalAid = pd.DataFrame(Total_total[i], columns=["Tot_Aid"])
                Run1_insurance = pd.concat([Run1_insurance, TotalAid.reindex(Run1_insurance.index)], axis=1)                  
                                

                AidGap = pd.DataFrame(GAP_total[i], columns=["TPR_Medicaid"])
                Run1_insurance = pd.concat([Run1_insurance, AidGap.reindex(Run1_insurance.index)], axis=1)

            Run1_insurance.to_csv("./results/Run1_TPR_insurance.csv")



    

if __name__ == '__main__':
    
    diseases = ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly',
           'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
           'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other',
           'Pneumonia', 'Pneumothorax', 'Support Devices']

    age_decile = ['60-80', '40-60', '20-40', '80-', '0-20']

    gender = ['M', 'F']
    race = ['WHITE', 'BLACK/AFRICAN AMERICAN',
            'HISPANIC/LATINO', 'OTHER', 'ASIAN',
            'AMERICAN INDIAN/ALASKA NATIVE']


    insurance = ['Medicare', 'Other', 'Medicaid']    
    
    TrueWithMeta = pd.read_csv("./True_withMeta.csv")
    pred = pd.read_csv("./results/bipred.csv")
    factor = [gender, age_decile, race, insurance]
    factor_str = ['gender', 'age_decile', 'race', 'insurance']
    
    for i in range(len(factor)):
        TPR_14(TrueWithMeta,pred, diseases, factor[i], factor_str[i]) 