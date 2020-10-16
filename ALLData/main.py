import torch
from train import *
from LearningCurve import *
from predictions import *
from plot import *
import pandas as pd
from Config import train_df, test_df, val_df
#----------------------------- q

diseases = ['No Finding', 'Atelectasis', 'Cardiomegaly',  'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation','Edema']
Age = ['60-80', '40-60', '20-40', '80-', '0-20']
Sex = ['M', 'F']

def main():

    MODE = "plot"  # Select "train" or "test", "Resume", "plot", "Threshold"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    val_df_size = len(val_df)
    print("Validation_df size:",val_df_size)
   
    train_df_size = len(train_df)
    print("Train_df size", train_df_size)
    
    test_df_size = len(test_df)
    print("Test_df size", test_df_size)


    if MODE == "train":
        ModelType = "densenet"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain( ModelType, CriterionType, device,LR)

        PlotLearnignCurve()


    if MODE =="test":
       
        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, device)


    if MODE == "Resume":
        ModelType = "Resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain( ModelType, CriterionType, device,LR)

        PlotLearnignCurve()

    if MODE == "plot":
        gt = pd.read_csv("./results/True.csv")
        pred = pd.read_csv("./results/bipred.csv")
        factor = [Sex, Age]
        factor_str = ['Sex', 'Age']



        # plot()
        for i in range(len(factor)):
            
        
            plot_sort_14( pred, diseases, factor[i], factor_str[i])


if __name__ == "__main__":
    main()
