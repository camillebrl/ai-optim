import pandas as pd
from os import listdir
from os.path import join
from os.path import isfile
import os
import zipfile
import seaborn as sns
from numpy import *
import math
import matplotlib.pyplot as plt
from pylab import *

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



names=["0.1_5e-05_ADAM_None_CosineAnnealingLR.csv","0.001_5e-05_ADAM_None_CosineAnnealingLR.csv",
        "0.1_5e-05_ADAM_None_ReduceLROnPlateau.csv","0.001_5e-05_ADAM_None_ReduceLROnPlateau.csv",
        "0.1_5e-05_SGD_0.9_CosineAnnealingLR.csv","0.001_5e-05_SGD_0.9_CosineAnnealingLR.csv",
        "0.1_5e-05_SGD_0.9_ReduceLROnPlateau.csv","0.001_5e-05_SGD_0.9_ReduceLROnPlateau.csv",
        "0.1_5e-05_SGD_0.95_CosineAnnealingLR.csv","0.001_5e-05_SGD_0.95_CosineAnnealingLR.csv",
        "0.1_5e-05_SGD_0.95_ReduceLROnPlateau.csv","0.001_5e-05_SGD_0.95_ReduceLROnPlateau.csv"        
]

labels = ['VGG11', 'VGG13', 'VGG16', 'VGG19']


model_path = "./ai-optim-master/lab1/results/"

for name in names:
    names_files=[]
    fig, ax = plt.subplots()
    fig2,ax2 = plt.subplots()
    max_acc_train={}
    max_acc_valid={}
    for label in labels:
        names_files.append(label+"_"+name)
    for name_file in names_files:
        df_model=pd.read_csv(model_path+name_file,delimiter=",").set_index("epoch")
        model=name_file[name_file.find("VGG"):name_file.find("_")]
        lr=name_file[name_file.find("0."):name_file.find("_5")]
        lr_diminution_function=name_file[name_file.rfind("_")+1:name_file.find(".csv")]
        if "SGD" in str(name_file):
            momentum=name_file[name_file.find("0.9"):name_file.rfind("_")]
            if model == "VGG11":
                color="lightsteelblue"
            if model == "VGG13":
                color="royalblue"
            if model == "VGG16":
                color="blue"
            if model == "VGG19":
                color="midnightblue"
            
            ax.set_title("représentation des modèles aux hyperparamètres: "+"méthode de descente de gradient SGD,"+" learning rate de "+lr+", scheduleer: "+lr_diminution_function+", momentum de "+momentum+", weight decay de 5e-5")
            ax.plot(df_model["validation_accuracy"],color,label=model)
            ax.set_ylabel="accuracy"
            ax.set_xlabel="epochs"
            ax.legend(labels)

            max_acc_valid[model]=max(df_model["validation_accuracy"])
            max_acc_train[model]=max(df_model["train_accuracy"])

        if "ADAM" in str(name_file):
            if model == "VGG11":
                color="lightsteelblue"
            if model == "VGG13":
                color="royalblue"
            if model == "VGG16":
                color="blue"
            if model == "VGG19":
                color="midnightblue"
            
            ax.set_title("représentation des modèles aux hyperparamètres: "+"méthode de descente de gradient ADAM,"+" learning rate de "+lr+", scheduler: "+lr_diminution_function+", weight decay de 5e-5")
            ax.plot(df_model["validation_accuracy"],color,label=model)
            ax.set_ylabel="accuracy"
            ax.set_xlabel="epochs"
            ax.legend(labels)

            max_acc_valid[model]=max(df_model["validation_accuracy"])
            max_acc_train[model]=max(df_model["train_accuracy"])


    for i in names_files:
        if "SGD" in i:
            method="SGD"
        
        if "ADAM" in str(i):
            method="ADAM"

    if method == "SGD":
        max_acc_hist_valid=[max_acc_valid["VGG11"],max_acc_valid["VGG13"],max_acc_valid["VGG16"],max_acc_valid["VGG19"]]
        x = np.arange(len(labels)) 
        width = 0.35
        ax2.set_title("représentation des modèles aux hyperparamètres: "+"méthode de descente de gradient SGD,"+" learning rate de "+lr+", scheduler: "+lr_diminution_function+", momentum de "+momentum+", weight decay de 5e-5")
        rects = ax2.bar(x - width/2, max_acc_hist_valid, width)
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)

        autolabel(rects,ax2)

        fig.tight_layout()

        plt.show()


    if method == "ADAM":
        max_acc_hist_valid=[max_acc_valid["VGG11"],max_acc_valid["VGG13"],max_acc_valid["VGG16"],max_acc_valid["VGG19"]]
        x = np.arange(len(labels)) 
        width = 0.35
        ax2.set_title("représentation des modèles aux hyperparamètres: "+"méthode de descente de gradient SGD,"+" learning rate de "+lr+", scheduler: "+lr_diminution_function+", weight decay de 5e-5")
        rects = ax2.bar(x - width/2, max_acc_hist_valid, width)
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)

        autolabel(rects,ax2)

        fig.tight_layout()

        plt.show()
