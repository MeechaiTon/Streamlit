import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import urllib3
import os
import csv
import json
import seaborn as sns
import datetime
from PIL import Image
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Classification in Machine Learning")

####################### Load Data #######################
iris = datasets.load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = pd.DataFrame(data=iris.target, columns=['target'])
df = pd.concat([iris_data,target], axis=1)

def pre_data() :
    ####################### Data Preparation #######################
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = 'target'), df['target'], test_size=0.50)
    return X_train, X_test, y_train, y_test

def ML(X_train, X_test, y_train, y_test) :
    ####################### Random Forest Algorithm #######################
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    clf_rf.predict(X_test)
    acc_rf = (y_test == clf_rf.predict(X_test)).sum() / len(y_test)
    conf_rf = confusion_matrix(y_test, clf_rf.predict(X_test), labels = target['target'].unique())

    ####################### K-Nearest Neighbors Algorithm #######################
    clf_knn = KNeighborsClassifier()
    clf_knn.fit(X_train, y_train)
    clf_knn.predict(X_test)
    acc_knn = (y_test == clf_knn.predict(X_test)).sum() / len(y_test)
    conf_knn = confusion_matrix(y_test, clf_knn.predict(X_test), labels = target['target'].unique())

    ####################### Support Vector Machine Algorithm #######################
    clf_svm = SVC(kernel='linear')
    clf_svm.fit(X_train, y_train)
    clf_svm.predict(X_test)
    acc_svm = (y_test == clf_svm.predict(X_test)).sum() / len(y_test)
    conf_svm = confusion_matrix(y_test, clf_svm.predict(X_test), labels = target['target'].unique())

    return acc_rf, acc_knn, acc_svm, conf_rf, conf_knn, conf_svm

####################### Process #######################
xtrain = []
xtest = []
ytrain = []
ytest = []

rf_acc = []
knn_acc = []
svm_acc = []

rf_conf = []
knn_conf = []
svm_conf = []

ML_result = []

for i in range(5) :
    set = pre_data()
    xtrain.append(set[0])
    xtest.append(set[1])
    ytrain.append(set[2])
    ytest.append(set[3])

    result_ML = ML(set[0], set[1], set[2], set[3])
    rf_acc.append(result_ML[0])
    knn_acc.append(result_ML[1])
    svm_acc.append(result_ML[2])
    rf_conf.append(result_ML[3])
    knn_conf.append(result_ML[4])
    svm_conf.append(result_ML[5])

ML_result.append(rf_acc)
ML_result.append(knn_acc)
ML_result.append(svm_acc)
####################### Layout Application #######################
st.markdown("<h1 style='text-align: center;'>Classification in Machine Learning </h1>", unsafe_allow_html=True)
st.write("Load Data")
library = '''import pandas as pd #pandas library for use dataframe
from sklearn import datasets #sklearn for load datasets
iris = datasets.load_iris() #load iris dataset
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names) #create iris measurement dataframe
target = pd.DataFrame(data=iris.target, columns=['target']) #create iris label dataframe
df = pd.concat([iris_data,target], axis=1) #prepare datafram for ML
print(df) '''
st.code(library, language='python')
st.dataframe(df,use_container_width=True)
st.write('For value in the target column is a species of iris')
st.write('''0, 1, and 2  is a 'setosa', 'versicolor' and 'virginica' respectively''')
st.markdown("***")

st.write("Data Preparation")
data_pre = '''from sklearn.model_selection import train_test_split #sklearn for prepare train and test data
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = 'target'), df['target'], test_size=0.50)
# X is a raw data, y is a data labels and use 50% of datasets as a training data'''
st.code(data_pre, language='python')

with st.expander("See Dataset"):
    ml_data = st.selectbox("Datasets", ['Training data', 'Testing data'], index=0)
    container1 = st.container()
    col1, col2= st.columns(2)
    for i in range(5) :
        if (ml_data == 'Training data') :
            with container1 :
                with col1 :
                    st.write("X " + ml_data + " " + str(i+1))
                    st.dataframe(xtrain[i].reset_index(drop=True),use_container_width=True)
                with col2 :
                    st.write("Y " + ml_data + " " + str(i+1))
                    st.dataframe(ytrain[i].reset_index(drop=True),use_container_width=True)
        if (ml_data == 'Testing data') :
            with container1 :
                with col1 :
                    st.write("X " + ml_data + " " + str(i+1))
                    st.dataframe(xtest[i].reset_index(drop=True),use_container_width=True)
                with col2 :
                    st.write("Y " + ml_data + " " + str(i+1))
                    st.dataframe(ytest[i].reset_index(drop=True),use_container_width=True)
st.markdown("***")

st.write("Machine Learning Part")
rf_code = '''# Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier #sklearn use RF
clf_rf = RandomForestClassifier() #RF model, can define hyperparameter of RF here
clf_rf.fit(X_train, y_train) #build RF model by using training data
clf_rf.predict(X_test) #use the model to predict testing data
acc_rf = (y_test == clf_rf.predict(X_test)).sum() / len(y_test) #get accuracy of RF model
# More information RF: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'''
st.code(rf_code, language='python')

knn_code = '''# K-Nearest Neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier #sklearn use KNN
clf_knn = KNeighborsClassifier() #KNN model, can define hyperparameter of KNN here
clf_knn.fit(X_train, y_train) #build KNN model by using training data
clf_knn.predict(X_test) #use the model to predict testing data
acc_knn = (y_test == clf_knn.predict(X_test)).sum() / len(y_test) #get accuracy of KNN model
# More information KNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'''
st.code(knn_code, language='python')

svm_code = '''# Support Vector Machine Algorithm
from sklearn.svm import SVC #sklearn use SVM
clf_svm = SVC(kernel='linear') #SVM model, can define hyperparameter of SVM here
clf_svm.fit(X_train, y_train) #build SVM model by using training data
clf_svm.predict(X_test) #use the model to predict testing data
acc_svm = (y_test == clf_svm.predict(X_test)).sum() / len(y_test) #get accuracy of SVM model
# More information SVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html'''
st.code(svm_code, language='python')

st.write("Confusion Matrix")
conf_code = conf_rf = '''from sklearn.metrics import confusion_matrix #sklearn library for confusion matrix
confusion_matrix(y_test, clf_rf.predict(X_test), labels = target['target'].unique()) #RF confusion matrix
conf_knn = confusion_matrix(y_test, clf_knn.predict(X_test), labels = target['target'].unique()) #KNN confusion matrix
conf_svm = confusion_matrix(y_test, clf_svm.predict(X_test), labels = target['target'].unique()) #SVM confusion matrix
'''
st.code(conf_code, language='python')
st.markdown("***")

st.markdown("<h5 style='text-align: center;'>Machine Learning Result (Running 5 rounds)</h5>", unsafe_allow_html=True)
container2 = st.container()
col_1, col_2, col_3 = st.columns(3)
with container2:
    with col_1:
        st.write("Random Forest")
        RF_result = pd.DataFrame(data=ML_result[0], columns=['Accuracy'])
        st.dataframe(RF_result,use_container_width=True)
    with col_2:
        st.write("K-Nearest Neighbors")
        KNN_result = pd.DataFrame(data=ML_result[1], columns=['Accuracy'])
        st.dataframe(KNN_result,use_container_width=True)
    with col_3:
        st.write("Support Vector Machine")
        SVM_result = pd.DataFrame(data=ML_result[2], columns=['Accuracy'])
        st.dataframe(SVM_result,use_container_width=True)
st.markdown("***")

st.markdown("<h5 style='text-align: center;'>Graph Visualization </h5>", unsafe_allow_html=True)
bar_fig = plt.figure(figsize=(14,5))
plt.boxplot(ML_result, notch = False, labels=['Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine'])
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
bar_fig
st.markdown("***")

st.markdown("<h5 style='text-align: center;'>Machine Learning Confusion Matrix </h5>", unsafe_allow_html=True)
container3 = st.container()
col_A, col_B, col_C, col_D, col_E = st.columns(5)
col = [col_A, col_B, col_C, col_D, col_E]

with container3 :
    for i in range(5) :
        with col[i] :
            st.write("Random Forest " + str(i+1))
            st.dataframe(rf_conf[i])
            st.write("K-Nearest Neighbors " + str(i+1))
            st.dataframe(knn_conf[i])
            st.write("Support Vector Machine " + str(i+1))
            st.dataframe(svm_conf[i])

st.markdown("************************************")
st.write("Made by : Meechai Homchan")
st.write("Date : 2022-10-06")
st.write("Markdown: Don't add Machine Learning Hyperparameter")
st.markdown("************************************")
