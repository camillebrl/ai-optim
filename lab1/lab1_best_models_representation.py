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


best_models_path="./ai-optim-master/lab1/best_models"
files = [join(best_models_path,f) for f in listdir(best_models_path) if isfile(join(best_models_path,f))]
for f in files:
    df_best_model=pd.read_csv(f,delimiter=",").set_index("epoch")
    print(f,max(df_best_model["validation_accuracy"]))

    fig,ax=plt.subplots()

    width = 0.35
    ax.set_title("représentation des meilleurs modèles:"+f[f.find("VGG"):f.find(".csv")])
    ax.plot(df_best_model["train_accuracy"],"r")
    ax.plot(df_best_model["validation_accuracy"],"b")
    ax.set_ylabel('Accuracy')
    ax.set_xlabel("Epochs")
    ax.legend(["train","test"])

    fig.tight_layout()

    plt.show()