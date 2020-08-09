# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:14:53 2020

@author: CarrieLai
"""

import numpy as np
import os
#from sklearn.model_selection import train_test_split

def load_all_feature(dir_save_feature, flag = 1):
    
    if flag == 1:
        
        feature = []
        label = []
        for i in range(1,4):
            fn_feature = "feature"+str(i)
            feature.append(np.load(dir_save_feature + fn_feature + ".npy"))
            save_label_npy = ".\\sample\\label" + str(i) + ".npy"
            label.append(np.load(save_label_npy))
        feature = np.concatenate(feature)
    #    feature = np.reshape(feature,[np.shape(feature)[0],np.shape(feature)[1]])
        label = np.concatenate(label)
    else:
        feature = np.load(dir_save_feature + "feature_2frame.npy" )
        label = np.load(dir_save_feature + "label_2frame.npy")
        
    return feature, label


def SeperateData(dir_save_feature,fn_dataset, flag = 1):

    if os.path.exists(dir_save_feature + fn_dataset[0] +".npy"):
        print(" =========  Train Set File is already exist  =========")
    if os.path.exists(dir_save_feature + fn_dataset[1] + ".npy"):
        print(" =========  Test Set File is already exist  =========")
    else:
        print(" >>>>>> Create Train Set and Test Set File !!!!!!  Start  !!!!!! ") 
        
        feature, label = load_all_feature(dir_save_feature, flag)
        
        ind_pos = np.array(np.where(label == 1))
        ind_pos = np.reshape(ind_pos, np.shape(ind_pos)[1])
        ind_neg = np.array(np.where(label == 0))
        ind_neg = np.reshape(ind_neg, np.shape(ind_neg)[1])
#        fea_pos = feature[ind_pos]
#        fea_neg = feature[ind_neg]
#        np.save(dir_save_feature + fn_feature_n_frame[0] + ".npy", fea_pos)
#        np.save(dir_save_feature + fn_feature_n_frame[1] + ".npy", fea_neg)
#        np.save(dir_save_feature + fn_feature_n_frame[2] + ".npy", ind_pos)
#        np.save(dir_save_feature + fn_feature_n_frame[3] + ".npy", ind_neg)
    
#        label_pos = np.uint8(np.ones(np.shape(fea_pos)[0]))
#        label_neg = np.uint8(np.zeros(np.shape(fea_neg)[0]))
#        X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(fea_pos, label_pos,test_size=0.2, random_state=0)
#        X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(fea_neg, label_neg,test_size=0.2, random_state=0)
#        X_train = np.array(list(X_train_pos) + list(X_train_neg))
#        X_test = np.array(list(X_test_pos) + list(X_test_neg))
#        y_train = np.array(list(y_train_pos) + list(y_train_neg))
#        y_test = np.array(list(y_test_pos) + list(y_test_neg))    
        
        ind_n = np.random.randint(0,np.shape(ind_neg)[0],np.shape(ind_neg)[0])
        #ind_p = np.random.randint(0,np.shape(ind_pos)[0],np.shape(ind_pos)[0])
        ind_p = ind_pos
        
        num_train_n = 60000
        #num_train_p =int(0.75 * np.shape(ind_p)[0])
        num_test_n = 20000
        if flag == 1:
            num_train_p = 846
        elif flag == 2:
            #ind_pos_original = np.load(dir_save_feature + "ind_pos.npy")
            num_train_p = 707
        else: #flag = 3
            num_train_n = 6000
            num_train_p = 70
        
        if flag <3:
            X_train = []
            X_train.extend(feature[ind_n[:num_train_n]])
            X_train.extend(feature[ind_p[:num_train_p]])
            X_train = np.array(X_train)
            y_train = np.array(list(np.uint8(np.zeros(num_train_n))) + list(np.uint8(np.ones(num_train_p))))

            X_test = []
            X_test.extend(feature[ind_n[num_train_n:(num_train_n+num_test_n)]])
            X_test.extend(feature[ind_p[num_train_p:]])
            X_test = np.array(X_test)
            y_test = np.array(list(np.uint8(np.zeros(num_test_n))) + list(np.uint8(np.ones(np.shape(ind_p)[0]-num_train_p))))
            np.save(dir_save_feature + fn_dataset[0] + ".npy", X_train)
            np.save(dir_save_feature + fn_dataset[2] + ".npy", y_train)
            np.save(dir_save_feature + fn_dataset[1] + ".npy", X_test)
            np.save(dir_save_feature + fn_dataset[3] + ".npy", y_test)
        else: #flag = 3
            X = np.load(dir_save_feature + "X_train_2frame.npy")
            y = np.load(dir_save_feature + "y_train_2frame.npy")
            X_train = []
            X_train.extend(X[:num_train_n])
            X_train.extend(X[num_train_n:num_train_n+num_train_p])
            y_train = []
            y_train.extend(y[:num_train_n])
            y_train.extend(y[num_train_n:num_train_n+num_train_p])
            np.save(dir_save_feature + fn_dataset[0] + ".npy", X_train)
            np.save(dir_save_feature + fn_dataset[2] + ".npy", y_train)           
        
        print(" >>>>>> Create Train Set and Test Set File !!!!!!  Done  !!!!!! ") 
        


def get_feature_2frame(dir_save_feature):

    if os.path.exists(dir_save_feature + "label_2frame.npy"):
        print(" =========  label_2frame File is already exist  =========")
    if os.path.exists(dir_save_feature + "feature_2frame.npy"):
        print(" =========  feature_2frame File is already exist  =========")
    else:
        feature, label = load_all_feature(dir_save_feature, flag = 1)
        ind = np.random.randint(0,np.shape(feature)[0]-1,np.shape(feature)[0]-1)
        
        feature_2frame = []
        label_2frame = []
        ind_p=[]
        for i in range(81000): 
            if sum(label[ind[i]:ind[i]+2]) < 2 :
                #f0 = np.reshape(feature[ind[i]:ind[i]+2],-1)
                #f0 = (f0 - min(f0))/(max(f0) - min(f0))
                feature_2frame.append(np.reshape(feature[ind[i]:ind[i]+2],-1))
                label_2frame.append(0)    
        feature_2frame = feature_2frame[:80000]
        label_2frame = label_2frame[:80000]

        for i in range(len(feature)-1): 
            if sum(label[i:i+2]) == 2:
                feature_2frame.append(np.reshape(feature[i:i+2],-1))
                label_2frame.append(1)
                ind_p.append(i)

        feature_2frame = np.array(feature_2frame)
        feature_2frame = np.reshape(feature_2frame,[np.shape(feature_2frame)[0],np.shape(feature_2frame[0])[0],1])
        np.save(dir_save_feature + "feature_2frame.npy", feature_2frame)
        np.save(dir_save_feature + "label_2frame.npy", label_2frame)
        np.save(dir_save_feature + "ind_pos.npy", ind_p)
    
    
def import_data_seperated(dir_save_feature,fn_dataset):
    
    X_train = np.load(dir_save_feature + fn_dataset[0] + ".npy")
    X_test = np.load(dir_save_feature + fn_dataset[1] + ".npy")
    y_train = np.load(dir_save_feature + fn_dataset[2] + ".npy")
    y_test = np.load(dir_save_feature + fn_dataset[3] + ".npy")    
    
    return X_train, X_test, y_train, y_test


def shuffle_data(X, y):
    ind = np.random.randint(0,np.shape(X)[0],np.shape(X)[0])
    X = X[ind]
    y = y[ind]
    return X,y
    
    