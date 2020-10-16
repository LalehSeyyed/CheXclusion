import torch
from train import *

from LearningCurve import *
from predictions import *
import pandas as pd
from plot import *

path_image = "/scratch/gobi2/projects/ml4h/datasets/CheXpert"


train_df_path ="/scratch/gobi2/projects/ml4h/datasets/CheXpert/split/July19/new_train.csv"
test_df_path ="/scratch/gobi2/projects/ml4h/datasets/CheXpert/split/July19/new_test.csv"
val_df_path = "/scratch/gobi2/projects/ml4h/datasets/CheXpert/split/July19/new_valid.csv"

# diseases = ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly',
#        'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
#        'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other',
#        'Pneumonia', 'Pneumothorax', 'Support Devices']


diseases = ['Lung Opacity',  'Atelectasis', 'Cardiomegaly',
            'Consolidation' , 'Edema',  'Enlarged Cardiomediastinum','Fracture',
            'Lung Lesion','No Finding',  'Pleural Effusion', 'Pleural Other','Pneumonia',
            'Pneumothorax', 'Support Devices' ]

# diseases = ['Lung Opacity',  'Atelectasis', 'Cardiomegaly',
#             'Consolidation' , 'Edema',  'Enlarged Cardiomediastinum','Fracture',
#             'Lung Lesion',  'Pleural Effusion', 'Pleural Other','Pneumonia',
#             'Pneumothorax', 'Support Devices' ]

# diseases = ['No Finding','Lung Lesion',
#        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
#        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
#        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
#        'Support Devices']
# Age = ['0-20', '20-40', '40-60', '60-80', '80-']
Age = ['60-80', '40-60', '20-40', '80-', '0-20']
gender = ['M', 'F']

def main():

    MODE = "test"  # Select "train" or "test", "Resume", "plot"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_df = pd.read_csv(val_df_path)
    val_df_size = len(val_df)
    print("Validation_df size:",val_df_size)

    train_df = pd.read_csv(train_df_path)
    train_df_size = len(train_df)
    print("Train_df size", train_df_size)
    
    test_df = pd.read_csv(test_df_path)
    test_df_size = len(test_df)
    print("Test_df size", test_df_size)


    if MODE == "train":
        ModelType = "densenet"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 5e-5

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device,LR)

        PlotLearnignCurve()


    if MODE =="test":
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)

        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, path_image, device)


    if MODE == "Resume":
        ModelType = "Resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.1e-3

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device,LR)

        PlotLearnignCurve()

    if MODE == "plot":
        gt = pd.read_csv("./results/True.csv")
        pred = pd.read_csv("./results/bipred.csv")
        factor = [gender, Age]
        factor_str = ['Sex', 'Age']
        for i in range(len(factor)):
            # plot_frequency(gt, diseases, factor[i], factor_str[i])
            # plot_TPR_CXP(pred, diseases, factor[i], factor_str[i])
             plot_sort_14(pred, diseases, factor[i], factor_str[i])
            # distance_max_min(pred, diseases, factor[i], factor_str[i])
            #plot_14(pred, diseases, factor[i], factor_str[i])
    # if MODE == "mean":
    #     pred = pd.read_csv("./results/bipred.csv")
    #     factor = [gender, age_decile]
    #     factor_str = ['Sex', 'Age']
    #     for i in range(len(factor)):
    #         mean(pred, diseases, factor[i], factor_str[i])

    # if MODE == "plot_14":
    #     pred = pd.read_csv("./results/bipred.csv")
    #     factor = [Age]
    #     factor_str = ['Age']
    #     for i in range(len(factor)):
    #         plot_14(pred, diseases, factor[i], factor_str[i])
    #         plot_Median(pred, diseases, factor[i], factor_str[i])


if __name__ == "__main__":
    main()
