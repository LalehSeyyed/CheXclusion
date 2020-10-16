import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Config import test_df
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


def plot_14(df, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
   # df = preprocess(df)
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

def plot_sort_14(Pred, diseases, category, category_name):

    df = test_df.merge(Pred, left_on='Jointpath', right_on='Jointpath')
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
   # df = df_copy
    plot_14(df, diseases, category, category_name)



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



