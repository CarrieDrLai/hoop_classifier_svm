# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:40:31 2020

@author: CarrieLai
"""


import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt



## libsvm

def convert_format(y):
    y = ((y - 0.5) * 2).astype(int)
    list_y = list(y)
    return list_y

def draw_roc(y_test, p_val, parameter, flag = 1):
    if flag == 1:
        package_name = "libsvm "
    elif flag == 2:
        package_name = "liblinear "
        
    fpr, tpr, threshold = metrics.roc_curve(y_test, np.array(p_val))
    np.save(".//roc//" + package_name + parameter + " Fea_2.npy", [tpr, fpr])
    color_list = ['red', 'blue', 'black', 'chocolate', 'yellow', 'green', 'pink', 'violet']
    plt.semilogx(fpr, tpr, color=color_list[1], label=parameter)
    plt.xlim([0.0, 10e-1])
    plt.xlabel('False Alarm Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    