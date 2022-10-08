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


warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")


####################### Load Data #######################
iris = datasets.load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = pd.DataFrame(data=iris.target, columns=['target'])
df = pd.concat([iris_data,target], axis=1)
df['species'] = 0
for i in range(len(df)) :
    df['species'][i] = iris.target_names[df['target'][i]]


####################### Scatter Plot #######################
axis = df.drop(columns=["target", "species"], axis=1).columns.tolist()
def scatter_plot(x_axis, y_axis, select) :
    scatter_fig = plt.figure(figsize=(8,6.5))
    scatter = df
    for i in range(len(select)) :
        if(select[i] == 0) :
            scatter = scatter.loc[(scatter['species'] != iris.target_names[i])]
    sns.scatterplot(data=scatter, x=x_axis, y=y_axis, hue="species")
    plt.title(str(x_axis[:-5].capitalize()) + " VS " + y_axis[:-5].capitalize())
    return scatter_fig

####################### Scatter Plot #######################
def bar_chart(stat) :
    bar_fig = plt.figure(figsize=(6,4))
    test = df.groupby("species").describe()
    step = 0
    for i in range(len(iris_data.columns)) :
        plt.bar(np.arange(len(test))[0]+step,test[(iris_data.columns[i], stat)][0],width =1,color ='r')
        plt.bar(np.arange(len(test))[1]+step,test[(iris_data.columns[i], stat)][1],width =1,color ='b')
        plt.bar(np.arange(len(test))[2]+step,test[(iris_data.columns[i], stat)][2],width =1,color ='g')
        step =step+4
    plt.legend(iris.target_names)
    plt.xticks([1,5,9,13],['sepal length', 'sepal width', 'petal length','petal width'])
    plt.grid()
    return bar_fig

####################### Distribution Plot #######################
def dis_plot(dis_x, dis_select) :
    dis_fig = plt.figure(figsize=(6.5,3.75))
    if (dis_select == 'all') :
        plt.title(str(dis_x[:-5].capitalize()) + ' Distribution')
        sns.kdeplot(data=df, multiple="stack", x=dis_x, hue='species')
    else :
        dis_data = df.loc[(df['species'] == dis_select)]
        sns.kdeplot(data=dis_data[dis_x], fill=True)
        plt.title(str(dis_select.capitalize()) + ' ' + str(dis_x[:-5]) + ' Distribution')
    return dis_fig

####################### Layout Application #######################
container0 = st.container()
colA, colB, colC = st.columns([1,3,1])
with container0:
    with colB:
        st.markdown("<h1 style='text-align: center;'>Iris Data Visualization</h1>", unsafe_allow_html=True)
        img = Image.open('iris.png')
        st.image(img, caption='Iris Species')
        st.write("The dataset consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor).")
        st.write("Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.")


container1 = st.container()
col1, col2, col3 = st.columns([2.5,1,2])
with container1:
    with col1:
        st.markdown("<h2 style='text-align: center;'>Iris Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(df.drop(columns = ['target']))
        st.write("Iris Dataset is a part of sklearn library. Sklearn comes loaded with datasets to practice machine learning techniques and iris is one of them.")
        st.write(" Iris has 4 numerical features and a tri class target variable. This dataset can be used for classification as well as clustering.")
    with col2:
        st.markdown("***")
        st.write("For Scatter Plot")
        x_axis = st.selectbox("X-Axis", axis, index=0)
        y_axis = st.selectbox("Y-Axis", axis, index=1)
        st.write('Species')
        A = st.checkbox(label = iris.target_names[0], value=True)
        B = st.checkbox(label = iris.target_names[1], value=True)
        C = st.checkbox(label = iris.target_names[2], value=True)
        select = [A,B,C]
        scatter_figure = scatter_plot(x_axis, y_axis, select)
    with col3:
        st.markdown("<h2 style='text-align: center;'>Scatter Plot</h2>", unsafe_allow_html=True)
        scatter_figure
        st.write("A scatter plot is a type of plot or mathematical diagram using Cartesian coordinates to display values for represent values for two different numeric variables.")

container2 = st.container()
col_1, col_2, col_3, col_4 = st.columns([0.2,1,0.5,1])
with container2:
    with col_1:
        st.markdown("***")
        st.write("For Bar Chart")
        stat = st.radio(label="Stat :", options=["mean","max","min","count","std","25%","50%","75%"], index=0)
        bar_figure = bar_chart(stat)
    with col_2:
        st.markdown("<h2 style='text-align: center;'>Bar Chart</h2>", unsafe_allow_html=True)
        bar_figure
        st.write("A bar chart is a graph which uses parallel rectangular shapes to represent changes in the size, value, or rate of something or to compare the amount of something relating to a number of different countries or groups.")
    with col_3:
        st.markdown("***")
        st.write("For Distribution Plot")
        dis_x = st.selectbox("Measurements", axis, index=0)
        dis_select = st.radio(label="Species :", options= np.append(iris.target_names,'all'), index=3)
        dis_figure = dis_plot(dis_x, dis_select)
    with col_4:
        st.markdown("<h2 style='text-align: center;'>Data Distribution</h2>", unsafe_allow_html=True)
        dis_figure
        st.write("Data distribution is a function that specifies all possible values for a variable and also quantifies the relative frequency (probability of how often they occur). Distributions are considered any population that has a scattering of data. ")
st.markdown("************************************")
st.write("Made by : Meechai Homchan")
st.write("Date : 2022-10-06")
st.markdown("************************************")
