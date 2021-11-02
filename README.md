# CheXclusion
# CheXclusion: Fairness gaps in deep chest X-ray classifiers

This is the code for the paper **'CheXclusion: Fairness gaps in deep chest X-ray classifiers' (https://arxiv.org/abs/2003.00827)** accepted in 'PSB 2021'.

In this paper, we examine the extent to which state-of-the-art deep learning classifiers trained to yield diagnostic labels from X-ray images are biased with respect to protected attributes, such as patient sex, age, race, and insurance type as a proxy for socioeconomic status. In particular, we examine the differences in true positive rate (TPR) across different subgroups per attributes. A high TPR disparity indicates that sick members of a protected subgroup would not be given correct diagnoses---e.g., true positives---at the same rate as the general population, even in an algorithm with high overall accuracy. 

We train convolution neural networks to predict 14 diagnostic labels in 3 prominent public chest X-ray datasets: MIMIC-CXR (MIMIC), Chest-Xray8 (NIH), CheXpert (CXP), as well as a multi-site aggregation of all those datasets (ALLData). 

**This code is also a good learning resource for researcher/students interested in training multi-label medical image pathology classifiers.** 

**Citation in Bibtex format:**

@article{CheXclusion_2020,
  title={CheXclusion: Fairness gaps in deep chest X-ray classifiers},  
  author={Seyyed-Kalantari, Laleh and Liu, Guanxiong and McDermott, Matthew and Chen, Irene and Marzyeh, Ghassemi},
  BOOKTITLE = {Pacific Symposium on Biocomputing},
  year={2021}
}

----------------------------------------------------------------------------------------------------------------------------
## Dataset access:
All three MIMIC-CXR, CheXpert, and ChestX-ray14 datasets used for this work are public under data use agreements. 

MIMIC-CXR dataset is available at: https://physionet.org/content/mimic-cxr/2.0.0/

CheXpert dataset is available at: https://stanfordmlgroup.github.io/competitions/chexpert/

ChestX-ray14 dataset is available at: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

Access to all three datasets requires user registration and the signing of a data use agreement. Only the MIMIC-CXR dataset requires the completion of an additional credentialing process. After following these procedures, the MIMIC-CXR data is available through PhysioNet (https://physionet.org/). The race/ethnicities and insurance type of the patients are not provided directly with the download of the MIMIC-CXR dataset. However, this data is available through merging the patient IDs in MIMIC-CXR with subject IDs in MIMIC-IV (https://physionet.org/content/mimiciv/0.4/) datasets, using the patient and admissions tables. Access to MIMIC-IV requires a similar procedure as MIMIC-CXR and the same credentialing process is applicable for both datasets. 


----------------------------------------------------------------------------------------------------------------------------
## Steps of runing code:
0 - Install the f1.yml environment. 

1 - Train your network using "MODE = train" in main.py --> the trained model will be saved as Checkpoint in results folder.

Note: We train 5 differenct model where all have the same hyper parameter and set up but they have different random seed. Finally, for all the results presented in the paper we average the results of 5 run and get the confidence interval (CI) and report them. Thus for the results we want to report such as AUCs, TPR disparities, FPRs, etc, when we are generating them it is essencial to remane them such that the name contain the run number. Later we use this naming protocol to gather and averaging the resilts over 5 run. At each section we wrote a guidline about what csv files needed to be raname and how.

2 - Test your network using "MODE = test" and runing main.py

The following csv files are generated: a) Eval.csv (contain AUC on validation set)  b)TestEval.csv (The AUC on test set for the model) c) True.csv (The true labels on Test set) d) preds.csv (The probability of each disease per image)  e) bipred.csv (The binary prediction of each label) f) Threshold.csv (The thereshold utilized to get binary predictions from probabilities per disease. It is calculated based on maximizing f1 score on validation set)

Rename TestEval.csv to Evel*.csv, where * is the number of run (e.g Evel1.csv for run1).

3 - Use the TrueDatawithMeta.ipynb to add metadata to the true labels of the test dataset and save it as True_withMeta.csv. This file and binary prediction bi_pred.csv of each result folder (associated to a random seed) are used to calculated TPRs.

4 - To get the actual TPR run the code Actual_TPR.py They are stored as: Run_TPR_race.csv Run_TPR_insurance.csv Run_TPR_age.csv Run_TPR_sex.csv

These files also contain the percentage of patient per subgroup/label and actual positive (p) and negative (-) cases per disease/subgroup in the original test dataset (True.csv). If there ia no patient per subgroup/lable the acciciated TPR is nan (These TPRs are later not considered at estimating the TPR disparities.)

Rename Run_TPRXX.csv to Run*_TPRXX.csv, where * is the number of run.(e.g Run1_TPR_sex.csv for run1).

5 - Select "MODE = "plot"" and run main.py It will produce the TPR disparity csv files.
TPR_Disparities_Age.csv TPR_Disparities_insurance.csv TPR_Disparities_race.csv TPR_Disparities_sex.csv

Rename them properly based on the run number. (e.g TPR_Disparities_Age.csv to TPR5_Disparities_Age.csv )

These files contain diseases labels,%of patient per subgroup, and the associated gap per label/subgroup. They will be used combined with 4 other run to calculate disparities considering the confidence intervals(CI). If a there is no patient for a subgroup then the TPR disparities are nan and they are not considered in TPR disparity canculation. The TPR disparites are caculated on subgroups that has member per disease/attribiute.

*** In "TPR_Disparities_race.csv" disease Px, PO, Co, LL, EC have 0 % American. *** In "TPR_Disparities_Age.csv" disease PO )% 0-20 have 0% members.

6 - rename the results forlder followed by the applied random seed for the checkpoint. (e.g. for random seed 31 use results31)

Do the step 2 to 6 for all 5 runs per dataset.

7 - create a folder and call it "results" to save the results of combining the 5 run.

8 - Run the Confidence.ipynb. It gives: a) Percentage of images per attribiute in whole data (test, train and validation). b) AUC performance with CI over 5 run.

9 - Confidence.py : plot the disparity figures of the 5 run including the CI using the csv fo step 5.

10 - CorrelationCoefficients.ipynb: Calculate the correlation coefficients of TPR sidparities and patient propotion per disease. It consider the Bonferroni correction significance level.

11 - SummaryTable.ipynb generate the values in summary table, which is table 3 in the paper

----------------------------------------------------------------------------------------------------------------------------
## Reproducing the results:
We have provided the Conda environment (f1.yml) in the same repository for reproducibility purposes. We are not able to share the trained model and the true label and predicted label CSV files of the test set due to the data-sharing agreement. However, we have provided the patient ID per test splits, random seed, and the code. Then, the true label and predicted label CSV files and trained models can be generated by users who have downloaded the data from the original source following the procedure that is described in the “Data availability” session.
