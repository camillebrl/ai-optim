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


def representation(dataset):
    model_binarized_path10=f"./{dataset}/0.1_binarized_thinet_with_dif_nb_param"
    files10 = [join(model_binarized_path10,f) for f in listdir(model_binarized_path10) if isfile(join(model_binarized_path10,f))]
    df_val10={}
    for f in files10:
        nom10=int(f[57:f.find("VGG")])
        df_model10=pd.read_csv(f,delimiter=",")
        df_val10[nom10]=df_model10["accuracy"][0]
    od_val10=collections.OrderedDict(sorted(df_val10.items()))
    #print(od_val10)
    df_final_val10=pd.DataFrame(od_val10,index=od_val10.keys())
    #print(df_final_val10)
    model_binarized_path20=f"./{dataset}/0.2_binarized_thinet_with_dif_nb_param"
    files20 = [join(model_binarized_path20,f) for f in listdir(model_binarized_path20) if isfile(join(model_binarized_path20,f))]
    df_val20={}
    for f in files20:
        nom20=int(f[57:f.find("VGG")])
        df_model20=pd.read_csv(f,delimiter=",")
        df_val20[nom20]=df_model20["accuracy"][0]
    od_val20=collections.OrderedDict(sorted(df_val20.items()))
    df_final_val20=pd.DataFrame(od_val20,index=od_val20.keys())

    fig, ax = plt.subplots()
    ax.set_title("représentation de l'accuracy en fonction du nb de paramètres avec binaryconnect")
    ax.plot(df_final_val10.iloc[2],"r")
    ax.plot(df_final_val20.iloc[2],"b")
    ax.set_ylabel('Accuracy')
    ax.set_xlabel("nb_of_parameters")
    ax.legend(["10 pourcent channels deleted","20 pourcent channels deleted"])
    fig.tight_layout()
    plt.show()


def representation2(dataset):
    fig, ax = plt.subplots()
    ax.set_ylabel('Accuracy')
    ax.set_xlabel("pruning_rate")
    couleurs=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'deeppink']
    ax.set_title("représentation de l'accuracy en fonction du nb de paramètres avec binaryconnect")
    model_binarized_path10=f"./{dataset}/0.1_binarized_thinet_with_dif_nb_param"
    files10 = [join(model_binarized_path10,f) for f in listdir(model_binarized_path10) if isfile(join(model_binarized_path10,f))]
    df_val10={}
    for f in files10:
        nom10=int(f[57:f.find("VGG")])
        df_model10=pd.read_csv(f,delimiter=",")
        df_val10[nom10]=df_model10["accuracy"][0]
    od_val10=collections.OrderedDict(sorted(df_val10.items()))
    #print(od_val10)
    # df_final_val10=pd.DataFrame(od_val10,index=od_val10.keys())
    #print(df_final_val10)
    model_binarized_path20=f"./{dataset}/0.2_binarized_thinet_with_dif_nb_param"
    files20 = [join(model_binarized_path20,f) for f in listdir(model_binarized_path20) if isfile(join(model_binarized_path20,f))]
    df_val20={}
    for f in files20:
        nom20=int(f[57:f.find("VGG")])
        df_model20=pd.read_csv(f,delimiter=",")
        df_val20[nom20]=df_model20["accuracy"][0]
    od_val20=collections.OrderedDict(sorted(df_val20.items()))
    # df_final_val20=pd.DataFrame(od_val20,index=od_val20.keys())

    ds = [od_val10, od_val20]
    d = {}
    for k in od_val10.keys():
        d[k] = tuple(d[k] for d in ds)
    print(d)

    df_final=pd.DataFrame.from_dict(d)
    print(df_final)
    #print(df_final)
    #print(df_final[:][0])
    
    legend=[]
    for i,col in enumerate(df_final.columns):
        print(df_final[col])
        ax.plot(df_final[col],couleurs[i])
        legend.append(col)
    
    ax.legend(legend)


    fig.tight_layout()
    plt.show()




representation2("minicifar")

# folders=["minicifar","cifar10","cifar100"]

# for folder in folders:
#     representation(folder)