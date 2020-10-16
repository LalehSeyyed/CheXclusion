import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random

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

ylabel = {'Age': 'AGE',
        'Sex': 'SEX',
        'F': 'FEMALE',
        'M': 'MALE'
        }

def plot_frequency(df, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
    df = preprocess(df)
    freq = []
    for d in diseases:
        cate = []
        for c in category:
            cate.append(len(df.loc[(df[d] == 1) & (df[category_name] == c), :]))
        freq.append(cate)
    freq = np.array(freq)
    if category_name == 'Age':
        plt.figure(figsize=(18,9))

        width = 0.075
    elif category_name == 'Sex':
        plt.figure(figsize=(18,9))

        width = 0.35
    ind = np.arange((len(diseases)))
    for i in range(len(category)):
        if category_name == 'Sex':
            plt.bar(ind + width * i, freq[:, i], width, label=ylabel[category[i]])
        else:
            plt.bar(ind + width * i, freq[:, i], width, label=category[i])
    plt.ylabel(str(category_name + ' FREQUENCY IN CXP').upper())
    plt.xticks(ind + width*(len(category)-1)/2, [diseases_abbr[k] for k in diseases])
    plt.legend()
    plt.savefig("./results/Frequency_" + category_name + ".pdf")


def tpr(df, d, c, category_name):
    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
    if len(gt) != 0:
        TPR = len(pred) / len(gt)
        return TPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return -1

def func(x, m, b):
    return m*x + b

#

def distance_max_min(df, diseases, category, category_name):
    df = preprocess(df)
    GAP_total = np.zeros((len(category), len(diseases)))
    percentage_total = np.zeros((len(category), len(diseases)))
    cate = []
    for c in range(len(category)):
        for d in range(len(diseases)):
            pred_disease = "bi_" + diseases[d]
            gt = df.loc[(df[diseases[d]] == 1) & (df[category_name] == category[c]), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 1) & (df[category_name] == category[c]), :]
            n_gt = df.loc[(df[diseases[d]] == 1) & (df[category_name] != category[c]) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 1) & (df[category_name] != category[c]) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[diseases[d]] == 1) & (df[category_name] == category[c]), :]
            pi_y = df.loc[(df[diseases[d]] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Sex':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, diseases[d], c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()
                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_total[c, d] = GAP
                percentage_total[c, d] = percentage
            else:
                GAP_total[c, d] = 51
                percentage_total[c, d] = 0
        
        category[c] = category[c].replace(' ', '_', 3)
        category[c] = category[c].replace('/', '_', 3)
        cate.append(category[c])
    
    df = pd.DataFrame(index=['Max', 'Min', 'Distance'])
    for d in range(len(diseases)):
        mask = GAP_total[:, d] < 50
        minimum = np.min(GAP_total[:, d][mask])
        maximum = np.max(GAP_total[:, d][mask])
        distance = maximum - minimum
        df[diseases[d]] = [maximum, minimum, distance]
    df.to_csv("./results/distance_max_min_" + category_name + ".csv")
    
def plot_TPR_CXP(df, diseases, category, category_name):

    plt.rcParams.update({'font.size': 18})
    df = preprocess(df)
    final = {}
    for c in category:
        result = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]
            
            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Sex':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                result.append([percentage, GAP])
            else:
                result.append([50, 50])
        result = np.array(result)
        plt.figure(figsize=(10, 8))
        plt.subplots_adjust()
        mask = result[:, 1] < 50
        plt.scatter(result[:, 0][mask], result[:, 1][mask], label='TPR', color='green')

        params, params_cov = curve_fit(func, result[:, 0][mask], result[:, 1][mask])
        plt.plot(result[:, 0][mask], func(result[:, 0][mask], params[0], params[1]), color='green')
        diseases = np.array(diseases)
        for d, x, y in zip(diseases[mask], result[:, 0][mask], result[:, 1][mask]):
            plt.annotate(diseases_abbr[d], color='green', xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
        plt.xlabel("% " + c)
        plt.ylabel("TPR " + ylabel[category_name] + " DISPARITY")
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        c = c.replace('>=', '_', 3)
        plt.savefig("./results/Median_TPR_" + category_name + "_" + c + ".pdf")
        
        ans = {'result': result,
                'mask': mask}
        final[c] = ans
    return final

def count(list1, l, r): 
    c = 0 
    # traverse in the list1 
    for x in list1: 
        # condition check 
        if x > l and x < r: 
            c+= 1 
    return c

def plot_14(df, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
    df = preprocess(df)
    GAP_total = []
    percentage_total = []
    cate = []

    print(diseases)



    if category_name == 'Sex':

        Run1_sex = pd.DataFrame(diseases,columns=["diseases"])

    if category_name == 'Age':

        Run1_age = pd.DataFrame(diseases,columns=["diseases"])


    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Sex':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        # print('Best Positive ' + c + ' ' + str(count(GAP_y, 0, 51)))
        # print('Worst Positive ' + c + ' ' + str(count(GAP_y, -50, 0)))
        # print('Zero ' + c + ' ' + str(GAP_y.count(0)))
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
        
    GAP_total = np.array(GAP_total)
    x = np.arange(len(diseases))
    fig = plt.figure(figsize=(18,9))
    ax = fig.add_subplot(111)
    for item in x:
        mask = GAP_total[:, item] < 50
        ann = ax.annotate('', xy=(item, np.max(GAP_total[:, item][mask])), xycoords='data',
                  xytext=(item, np.min(GAP_total[:, item][mask])), textcoords='data',
                  arrowprops=dict(arrowstyle="<->",
                                  connectionstyle="bar"))
        
    for i in range(len(GAP_total)):
        s = np.multiply(percentage_total[i],1000)
        mask = GAP_total[i] < 50
        plt.scatter(x[mask], GAP_total[i][mask], s=s, marker='o', label=cate[i])

        print("Perc", percentage_total[i])
        print("GAPt", GAP_total[i][mask])
        
        if category_name == 'Age':



            if i == 0:
                Percent6 = pd.DataFrame(percentage_total[i], columns=["%60-80"])
                Run1_age = pd.concat([Run1_age, Percent6.reindex(Run1_age.index)], axis=1)

                Gap6 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_60-80"])
                Run1_age = pd.concat([Run1_age, Gap6.reindex(Run1_age.index)], axis=1)

            if i== 1:
                 Percent4 = pd.DataFrame(percentage_total[i], columns=["%40-60"])
                 Run1_age = pd.concat([Run1_age, Percent4.reindex(Run1_age.index)], axis=1)

                 Gap4 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_40-60"])
                 Run1_age = pd.concat([Run1_age, Gap4.reindex(Run1_age.index)], axis=1)

            if i == 2:
                Percent4 = pd.DataFrame(percentage_total[i], columns=["%20-40"])
                Run1_age = pd.concat([Run1_age, Percent4.reindex(Run1_age.index)], axis=1)

                Gap4 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_20-40"])
                Run1_age = pd.concat([Run1_age, Gap4.reindex(Run1_age.index)], axis=1)

            if i == 3:
                Percent8 = pd.DataFrame(percentage_total[i], columns=["%80-"])
                Run1_age = pd.concat([Run1_age, Percent8.reindex(Run1_age.index)], axis=1)

                Gap8 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_80-"])
                Run1_age = pd.concat([Run1_age, Gap8.reindex(Run1_age.index)], axis=1)

            if i == 4:
                Percent0 = pd.DataFrame(percentage_total[i], columns=["%0-20"])
                Run1_age = pd.concat([Run1_age, Percent0.reindex(Run1_age.index)], axis=1)

                Gap0 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_0-20"])
                Run1_age = pd.concat([Run1_age, Gap0.reindex(Run1_age.index)], axis=1)

            Run1_age.to_csv("./results/Run1_Age.csv")


        if category_name == 'Sex':

            if i == 0:
                MalePercent = pd.DataFrame(percentage_total[i], columns=["%M"])
                Run1_sex = pd.concat([Run1_sex, MalePercent.reindex(Run1_sex.index)], axis=1)

                MaleGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_M"])
                Run1_sex = pd.concat([Run1_sex, MaleGap.reindex(Run1_sex.index)], axis=1)

            else:
                FeMalePercent = pd.DataFrame(percentage_total[i], columns=["%F"])
                Run1_sex = pd.concat([Run1_sex, FeMalePercent.reindex(Run1_sex.index)], axis=1)

                FeMaleGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_F"])
                Run1_sex = pd.concat([Run1_sex, FeMaleGap.reindex(Run1_sex.index)], axis=1)

            Run1_sex.to_csv("./results/Run1_sex.csv")


        
        


    plt.xticks(x, [diseases_abbr[k] for k in diseases])
    plt.ylabel("TPR " + ylabel[category_name] + " DISPARITY")
    plt.legend()
    plt.savefig("./results/Median_Diseases_x_GAP_" + category_name + ".pdf")

def plot_sort_14(df, diseases, category, category_name):
    df_copy = df
    df = preprocess(df)
    GAP_total = []
    percentage_total = []
    cate = []
    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Sex':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
        
    GAP_total = np.array(GAP_total)
    percentage_total = np.array(percentage_total)

    difference = {}
    for i in range(GAP_total.shape[1]):
        mask = GAP_total[:, i] < 50
        difference[diseases[i]] = np.max(GAP_total[:, i][mask]) - np.min(GAP_total[:, i][mask])
    sort = [(k, difference[k]) for k in sorted(difference, key=difference.get, reverse=False)]
    diseases = []
    for k, _ in sort:
        diseases.append(k)
    df = df_copy
    plot_14(df, diseases, category, category_name)

def preprocess(split):
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/CheXpert/map.csv")
    if 'Atelectasis' in split.columns:
        details = details.drop(columns=['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices'])
    split = split.merge(details, left_on="Path", right_on="Path")
    split['Age'] = np.where(split['Age'].between(0,19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20,39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40,59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60,79), 79, split['Age'])
    split['Age'] = np.where(split['Age']>=80, 81, split['Age'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81, 'Male', 'Female'], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-", 'M', 'F'])
    return split

def random_split(map_path, total_subject_id, split_portion):
    df = pd.read_csv(map_path)
    subject_df = pd.read_csv(total_subject_id)
    subject_df['random_number'] = np.random.uniform(size=len(subject_df))


    train_id = subject_df[subject_df['random_number'] <= split_portion[0]]
    valid_id = subject_df[(subject_df['random_number'] > split_portion[0]) & (subject_df['random_number'] <= split_portion[1])]
    test_id = subject_df[subject_df['random_number'] > split_portion[1]]

    train_id = train_id.drop(columns=['random_number'])
    valid_id = valid_id.drop(columns=['random_number'])
    test_id = test_id.drop(columns=['random_number'])

    train_id.to_csv("train_id.csv", index=False)
    valid_id.to_csv("valid_id.csv", index=False)
    test_id.to_csv("test_id.csv", index=False)

    train_df = train_id.merge(df, left_on="subject_id", right_on="subject_id")
    valid_df = valid_id.merge(df, left_on="subject_id", right_on="subject_id")
    test_df = test_id.merge(df, left_on="subject_id", right_on="subject_id")

    print(len(train_df))
    print(len(valid_df))
    print(len(test_df))

    train_df.to_csv("new_train.csv", index=False)
    valid_df.to_csv("new_valid.csv", index=False)
    test_df.to_csv("new_test.csv", index=False)

# random_split("map.csv", "total_subject_id_with_gender.csv", [0.8, 0.9])



