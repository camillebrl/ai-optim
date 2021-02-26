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
import collections


def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# model_path="./models_with_dif_nb_param"
# files = [join(model_path,f) for f in listdir(model_path) if isfile(join(model_path,f))]

# df_train={}
# df_val={}
# for f in files:
#     nom=int(f[27:f.find("_VGG")])
#     df_model=pd.read_csv(f,delimiter=",").set_index("epoch")
#     df_train[nom]=max(df_model["train_accuracy"])
#     df_val[nom]=max(df_model["validation_accuracy"])
# od_train=collections.OrderedDict(sorted(df_train.items()))
# od_val=collections.OrderedDict(sorted(df_val.items()))
# df_final_train=pd.DataFrame(od_train,index=od_train.keys())
# df_final_val=pd.DataFrame(od_val,index=od_val.keys())
# fig, ax = plt.subplots()
# ax.set_title("représentation de l'accuracy en fonction du nb de paramètres avec des poids en full precision")
# ax.plot(df_final_train.iloc[2],"g")
# ax.plot(df_final_val.iloc[2],"r")
# ax.set_ylabel('Accuracy')
# ax.set_xlabel("nb_of_parameters")
# ax.legend(["train","validation"])
# fig.tight_layout()
# plt.show()


def representation(folder):
    model_binarized_path=f"./{folder}/models_binarized_with_dif_nb_param"
    files = [join(model_binarized_path,f) for f in listdir(model_binarized_path) if isfile(join(model_binarized_path,f))]
    df_train={}
    df_val={}
    for f in files:
        nom=int(f[47:f.find("_VGG")])
        df_model=pd.read_csv(f,delimiter=",").set_index("epoch")
        df_train[nom]=max(df_model["train_accuracy"])
        df_val[nom]=max(df_model["validation_accuracy"])
    od_train=collections.OrderedDict(sorted(df_train.items()))
    od_val=collections.OrderedDict(sorted(df_val.items()))
    df_final_train=pd.DataFrame(od_train,index=od_train.keys())
    df_final_val=pd.DataFrame(od_val,index=od_val.keys())
    fig, ax = plt.subplots()
    ax.set_title("représentation de l'accuracy en fonction du nb de paramètres avec binaryconnect")
    ax.plot(df_final_train.iloc[2],"g")
    ax.plot(df_final_val.iloc[2],"r")
    ax.set_ylabel('Accuracy')
    ax.set_xlabel("nb_of_parameters")
    ax.legend(["train","validation"])
    fig.tight_layout()
    plt.show()


folders=["minicifar","cifar10","cifar100"]

for folder in folders:
    representation(folder)