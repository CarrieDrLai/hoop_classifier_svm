# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:44:23 2020

@author: CarrieLai
"""

# from get_dataset import GetHoopPos,GetDataset,create_annotation
from load_dataset import load_dataset
# from extract_feature import extract_feature,extract_feature_using_package
# from lr_msgd import MSGD_LogisticRegression
from data_separate import SeperateData, get_feature_2frame, import_data_seperated, shuffle_data
from lib_using import convert_format, draw_roc
#from naive_classifier import Predict

import os
import matplotlib.pyplot as plt

#############################   Read Congig File   #############################

print(" ============== Read Config File ==============\n")
f = open('config.txt','r')
config = f.readlines()
f.close()

var_list=[]
for var in config:
    var = var.strip()
    var_name = var.split("=")[0].strip()
    var_value = var.split("=")[1].strip()
    if var_name == 'crop_size' or var_name == 'colume':
        var_value = eval(var_value)
    if var_name == 'block_size' or var_name == 'block_stride' or var_name == 'cell_size' or var_name == 'bin_num':
        var_value = eval(var_value)
    if var_name == 'training_set' or var_name == 'epoch' or var_name == 'learning_rate' or var_name == 'batch_size' or var_name == 'lamda' or var_name == 'improve_thresh':
        var_value = eval(var_value)    
    if var_name == 'frame' or var_name == 'line_search' or var_name == 'weight_decay' or var_name == 'early_stop':
        var_value = eval(var_value) 
    if var_name == 'c' or var_name == 't' or var_name == 'g' or var_name == 'd':
        var_value = eval(var_value) 
    var_list.append([var_name, var_value])
    print( " " + var_name + " = " + str(var_value))
    
var_list = dict(var_list)
    
print(" ============== Read Config File Success ==============\n")

#############################        Parameter        ###################################


fn = var_list["fn"]
fn_data = fn + ".jpg"

crop_size = var_list["crop_size"]
colume = var_list["colume"]
fn_dataset = ["X_train", "X_test", "y_train", "y_test"]

#thresh = var_list["thresh"]
task_type = var_list["task_type"]
# 1. locate: get HoopPos
# 2. crop : get all patch, save data.jpg and rough separete(and then separate manually)  
# 3. annotation
# 4. feature 
# 5. classification


#############################  Step 1 : Prepare Dataset  ################################

if task_type == 'locate': 

    fn_video = var_list["fn_video"]
    dir_video = ".\\video\\" + fn_video
    
    ################  (a)Get Hoop Position
    hoop = GetHoopPos(fn_video,crop_size)
    hoop_pos = hoop.get_pos()        #hoop_pos = [(924,133)]

elif task_type == 'crop':
    
    fn_video = var_list["fn_video"]
    dir_video = ".\\" + fn_video
    path_data = ".\\sample\\" + fn_data
    save_data_npy = ".\\sample\\" + fn + ".npy"

    dir_pos = var_list["dir_pos"]
    dir_neg = var_list["dir_neg"]
    
    ################   (b)Get all data & save data as npy
    
    Data = GetDataset(fn_video, path_data, save_data_npy, dir_pos, dir_neg, crop_size,hoop_pos,colume)  
    patch_all = Data.get_data()
    Data.rough_separate()
    ################  (c)Manually seperate the data after rough seperation 
    
elif task_type == 'annotation':
    ################  (d)Make Annotation File
    fn_annotation = var_list["fn_annotation"]
    path_annotation = ".\\sample\\" + fn_annotation
    
    save_data_npy = ".\\sample\\" + fn + ".npy"
    save_label_npy = ".\\sample\\label" + fn[-1] + ".npy"

    dir_pos = var_list["dir_pos"]
    dir_neg = var_list["dir_neg"]

    create_annotation(dir_pos, dir_neg, path_annotation, save_label_npy, save_data_npy, crop_size, colume)

###########################    Step 2 : Extract Feature   ##############################

elif task_type == 'feature':

    path_data = ".\\sample\\" + fn_data
    fn_annotation = var_list["fn_annotation"]
    path_annotation = ".\\sample\\" + fn_annotation
    fn_feature = var_list["fn_feature"]
    
    block_size = var_list["block_size"]
    block_stride = var_list["block_stride"]
    cell_size = var_list["cell_size"]
    bin_num = var_list["bin_num"]
    
    ###############  (a)load dataset

    Dataset = load_dataset(path_annotation,path_data,crop_size)
    data, label = Dataset.load_data() 

    ###############  (b)Extract Feature  

#    hog = extract_feature(dir_save_feature, fn_feature, data, block_size, block_stride, cell_size, bin_num)
#    feature = hog.HoG_output_vector()
    
    feature = extract_feature_using_package(data,dir_save_feature,fn_feature)


elif task_type == 'seperate':
    
    dir_save_feature = var_list["dir_save_feature"]
    
    fn_dataset = ["X_train", "X_test", "y_train", "y_test"]
    SeperateData(dir_save_feature, fn_dataset,flag = 1)

    fn_dataset_2frame = ["X_train_2frame", "X_test_2frame", "y_train_2frame", "y_test_2frame"]
    get_feature_2frame(dir_save_feature)
    SeperateData(dir_save_feature,fn_dataset_2frame,flag = 2)
    
    fn_dataset_cross = ["X_train_part", "X_test_part", "y_train_part", "y_test_part"]
    SeperateData(dir_save_feature,fn_dataset_cross,flag = 3)
    

elif task_type == 'classification':
    
    dir_save_feature = var_list["dir_save_feature"]

    ##############  (c)Fit Model & Predict  

    fn_dataset_2frame = ["X_train_2frame", "X_test_2frame", "y_train_2frame", "y_test_2frame"]
    
#    X_train, X_test, y_train, y_test = import_data_seperated(dir_save_feature,fn_dataset)
    X_train, X_test, y_train, y_test = import_data_seperated(dir_save_feature,fn_dataset_2frame)
    # after normalization
    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test)
    
    
####################################
####### Logistic Regression
####################################

#    training_set = var_list["training_set"]
#    n_epochs = var_list["epoch"]
#    lr = var_list["learning_rate"]
#    batch_size = var_list["batch_size"]
#    lamda = var_list["lamda"]
#    improve_thresh = var_list["improve_thresh"]
#
#    frame = var_list["frame"]
#    line_search = var_list["line_search"]
#    weight_decay = var_list["weight_decay"]
#    early_stop = var_list["early_stop"]    
#
#    set_num = [1000, 2000, 4000, 32000, 48000, 60000]
#    for i in range(6):
#        training_set = set_num[i]
#    idx = np.random.randint(0,np.shape(X_train)[0],training_set)
#    X_train_part = X_train[idx]
#    y_train_part = y_train[idx]
##    
#    model = MSGD_LogisticRegression(X_train, y_train, X_test, y_test, lr, batch_size, n_epochs, lamda, improve_thresh, early_stop,weight_decay, n_class=2)
##    model.load_model()
#    model.msgd_optimization()
#    model.predict()
#    model.draw_ROC()
#    model.save_model()
    
####################################
####### libsvm
####################################

    from libsvm import svmutil
    from svmutil import *
    import gc
    c = var_list["c"]  
    t = var_list["t"]   # 0: linear; 1 : polynomial; default : 2, rbf kernel
    #g = var_list["g"]   #gamma
    g = [0.001,0.01,0.1]
    #d = var_list["d"]   #degree
    print(" >>>> Convert y format !!!!")
    y_train = convert_format(y_train)
    y_test = convert_format(y_test)
    
    for i in range(len(g)):
        
        #parameter = "-s 0" + " -t " + str(t) + " -c " + str(c)
        parameter = "-s 0" + " -t " + str(t) + " -c " + str(c) + " -g " + str(g[i])
        #parameter = "-s 0" + " -t " + str(t) + " -c " + str(c) + " -d " + str(d) + " -g " + str(g)
        
        if os.path.exists(".\\roc\\libsvm " + parameter + " Fea_2.npy"):
            print(" >>>> File is already exist.")
            #continue
        else:
            print(" >>>> Get parameter !!!!")
            param = svm_parameter(parameter)
            prob = svm_problem(y_train, X_train)
            
            print(" >>>> Start Training !!!!")
            model = svm_train(prob, param)
            svm_save_model('model/' + parameter + '.model', model)
            print(" >>>> Training Done !!!!")
        #    del y_train, x_train, prob
        #    gc.collect()
        
            print(" >>>> Start Testing !!!!")
            p_label, p_acc, p_val = svm_predict(y_test, X_test, model)
            # print(p_acc)
            print(parameter + ' acc = %f' % p_acc[0])
            draw_roc(y_test, p_val, parameter, flag = 1)



####################################
####### liblinear
####################################

#    import liblinear
#    import liblinearutil as util
#    import liblinear_commonutil
#    
#    y_train = convert_format(y_train)
#    y_test = convert_format(y_test)
#    #parameter = '-c 0.5 -v 5'
#    c = [0.5, 1]
#    for i in range(len(c)):
#        parameter = '-c ' + str(c[i])
#        if os.path.exists(".\\roc\\liblinear " + parameter + " Fea_2.npy"):
#            print(" >>>> File is already exist.")
#        else:
#            print(" >>>> Get parameter !!!!")
#            
#            para = util.parameter(parameter)
#            prob = util.problem(y_train,X_train)
#        
#            print(" >>>> Start Training !!!!")
#            model = util.train(prob, para)
#            print(" >>>> Training Done !!!!")
#            # m = train(y, x, '-c 5')
#            # m = train(prob, '-w1 5 -c 5')
#            # m = train(prob, param)
#            # CV_ACC = train(y, x, '-v 3')
#        
#            p_label, p_acc, p_val = util.predict(y_test, X_test, model)
#            print(parameter + ' acc = %f' % p_acc[0])
#            draw_roc(y_test, p_val, parameter, flag = 2)


