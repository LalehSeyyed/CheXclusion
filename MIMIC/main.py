import torch
from train import *
from LearningCurve import *
from predictions import *
from TPR_Disparity import *
import pandas as pd

#----------------------------- q
path_image = "/scratch/gobi2/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/"


train_df_path ="/scratch/gobi2/projects/ml4h/datasets/new_split/8-1-1/new_train.csv"
test_df_path ="/scratch/gobi2/projects/ml4h/datasets/new_split/8-1-1/new_test.csv"
val_df_path = "/scratch/gobi2/projects/ml4h/datasets/new_split/8-1-1/new_valid.csv"
# we use MIMIC original validation dataset as our new test dataset and the new_test.csv as out validation dataset

diseases = ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']
# diseases = ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly',
#        'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
#        'Lung Lesion', 'Pleural Effusion', 'Pleural Other',
#        'Pneumonia', 'Pneumothorax', 'Support Devices']
age_decile = ['60-80', '40-60', '20-40', '80-', '0-20']

gender = ['M', 'F']
race = ['WHITE', 'BLACK/AFRICAN AMERICAN',
        'HISPANIC/LATINO', 'OTHER', 'ASIAN',
        'AMERICAN INDIAN/ALASKA NATIVE']
# race = ['WHITE', 'BLACK/AFRICAN AMERICAN', 'ASIAN',
#         'AMERICAN INDIAN/ALASKA NATIVE']

insurance = ['Medicare', 'Other', 'Medicaid']





def main():

    MODE = "plot"  # Select "train" or "test", "Resume", "plot", "Threshold"

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
        LR = 0.5e-3

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
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device,LR)

        PlotLearnignCurve()

    if MODE == "plot":
        TrueWithMeta = pd.read_csv("./True_withMeta.csv")
        pred = pd.read_csv("./results/bipred.csv")
        factor = [gender, age_decile, race, insurance]
        factor_str = ['gender', 'age_decile', 'race', 'insurance']



        # plot()
        for i in range(len(factor)):
            
            #plot_frequency(gt, diseases, factor[i], factor_str[i])
            
            TPR_Disparities(TrueWithMeta, pred, diseases, factor[i], factor_str[i])
           

        
if __name__ == "__main__":
    main()
