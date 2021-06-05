import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import modul as md

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# st.title("Машин сургалтын аргад тулгуурлан хэрэглэгчдийг сегментчилэх")
#RFM analyze
st.sidebar.header('RFM анализ')
def main():
    st.sidebar.header('Сургалт')

    dataset_name = st.sidebar.selectbox("Өгөгдөл сонгох", ("Customer segmentation", "Product segmentation"))
    ml_alogs = st.sidebar.selectbox("Машин сургалтын аргууд", ('Decision Tree', 'Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Naive Bayes',"K-Means"))


    X,y,df = md.get_df(dataset_name)


    st.write("**Сонгосон өгөгдлийн багц**",df)
    st.write("**Өгөгдлийн хэмжээ** :", X.shape)
    st.write("**Бүлгийн тоо** :", len(np.unique(y)))

    #Data beldelt
    x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    y_real = np.concatenate((y_train, y_test))

    # Surgaltiin arguud
    def select_ml_alog(ml_alogs):
        if ml_alogs == "Decision Tree":
            md.dTree(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "Random Forest":
            md.rForest(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "Logistic Regression":
            md.lRegression(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "Support Vector Machine":
            md.SVM(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "Naive Bayes":
            md.nBayes(x_train, y_train, x_test, y_test, st)

        elif ml_alogs == "K-Means":
            k = st.sidebar.slider("K", 2, 10)
            md.kMeans(X, k, df, st)

    select_ml_alog(ml_alogs)




filename = st.sidebar.text_input('Enter a file path:')
try:
    with open(filename) as input:
        md.rfm(filename,st)
        main()
except FileNotFoundError:
    st.error('fail oldsongui')




