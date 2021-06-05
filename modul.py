import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from streamlit.elements.deck_gl_json_chart import EMPTY_MAP
#RFM 
def rfm(filename,st):
    mdata = pd.read_csv(filename)
    st.dataframe(mdata)
    st.write(mdata.shape)
    mdata['Order Date'] = pd.to_datetime(mdata['Order Date'], format='%Y-%m-%d')


    form = st.sidebar.form(key='my_form')
    form.subheader('Talbar songoh')
    date_col = form.selectbox("Hudaldan avaltiin ognoo", (mdata.columns))
    order_id_col = form.selectbox("Hudaldan avaltiin dugaar", (mdata.columns))
    total_col = form.selectbox("Niit dun", (mdata.columns))
    submit_button = form.form_submit_button(label='Bolson')
    df_RFM = mdata.groupby('Customer ID').agg({date_col: lambda y: (mdata[date_col].max().date() - y.max().date()).days,
                                        order_id_col: lambda y: len(y.unique()),  
                                        total_col: lambda y: round(y.sum(),2)})
    if(len(df_RFM.columns)>2):
        df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
        df_RFM = df_RFM.sort_values('Monetary', ascending=False)
        # st.dataframe(df_RFM)
        # st.write(df_RFM.shape)

    else:
        st.error('baganuudaa songono uu')

    quantiles = df_RFM.quantile(q=[0.5])
    df_RFM['R']=np.where(df_RFM['Recency']<=int(quantiles.Recency.values), 2, 1)
    df_RFM['F']=np.where(df_RFM['Frequency']>=int(quantiles.Frequency.values), 2, 1)
    df_RFM['M']=np.where(df_RFM['Monetary']>=int(quantiles.Monetary.values), 2, 1)
    # st.dataframe(df_RFM)
    # st.write(df_RFM.shape)

    df_RFM.loc[(df_RFM['R']==2) & (df_RFM['M']==2),'class'] = 1
    df_RFM.loc[(df_RFM['R']==1) & (df_RFM['M']==2),'class'] = 2
    df_RFM.loc[(df_RFM['R']==2) & (df_RFM['M']==1),'class'] = 3
    df_RFM.loc[(df_RFM['R']==1) & (df_RFM['M']==1),'class'] = 4
    df_RFM['class']=df_RFM['class'].astype(int)
    # st.dataframe(df_RFM)
    # st.write(df_RFM.shape)

    result = pd.merge(mdata, df_RFM, on="Customer ID")


    train_data=result[['Customer ID','Recency','Frequency','Monetary','class']].copy()

    customers = train_data['Customer ID'].unique()
    customer_df=pd.DataFrame(customers, columns=['Customer ID'])

    train_data_merged = pd.merge(left=customer_df, right=train_data, left_on='Customer ID', right_on='Customer ID')

    train_data_merged=train_data_merged.drop_duplicates()
    train_data_merged = train_data_merged.reset_index(drop=True)
    st.dataframe(train_data_merged)
    st.write(train_data_merged.shape)

# data beldelt
def get_df(dataset_name):
    if dataset_name == "Customer segmentation":
        df = pd.read_csv("train_data_customer_segmentation.csv")
    else:
        df = pd.read_csv("train_data_product_segmentation.csv")
    X = df.iloc[:, 1:4].values 
    y = df.iloc[:, 4].values
    return X,y,df

#Decision Tree
def dTree(x_train, y_train, x_test, y_test, st):
    model = DecisionTreeClassifier() 
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    dt_train_score = model.score(x_train, y_train)
    dt_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(dt_test_score*100,2),"%")

#Random Forest
def rForest(x_train, y_train, x_test, y_test, st):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    rf_train_score = model.score(x_train, y_train)
    rf_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(rf_test_score*100,2),"%")


#Logistic Regression
def lRegression(x_train, y_train, x_test, y_test, st):
    model = LogisticRegression()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    lr_train_score = model.score(x_train, y_train)
    lr_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(lr_test_score*100,2),"%")

#Support Vector Machine
def SVM(x_train, y_train, x_test, y_test, st):
    model = SVC()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    svm_train_score = model.score(x_train, y_train)
    svm_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(svm_test_score*100,2),"%")

#Naive Bayes
def nBayes(x_train, y_train, x_test, y_test, st):
    model = GaussianNB()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    nb_train_score = model.score(x_train, y_train)
    nb_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл** :",round(nb_test_score*100,2),"%")

#Kmeans
def kMeans(X, k, df, st):
    scaler = StandardScaler()
    scaler.fit(X)
    selected_data_std=scaler.transform(X)
    selected_data_std_df=pd.DataFrame(selected_data_std, columns=['Recency','Frequency','Monetary'])
    matrix = selected_data_std_df.to_numpy()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(selected_data_std_df)
    plt.hist(kmeans.labels_)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    cluster_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_data_std_df.columns)

    cluster_df_std = pd.DataFrame(kmeans.cluster_centers_, columns=selected_data_std_df.columns)
    
    selected_data_with_X = df.iloc[:,1:4] 
    selected_data_with_X['cluster'] = kmeans.labels_
    for i in range(k):
        selected_data_with_X.loc[selected_data_with_X['cluster'] == i].value_counts(normalize=True)

    pca = PCA(n_components=3)
    pca.fit(selected_data_std)
    pca.components_
    X1 = pd.DataFrame(pca.components_, columns =['Recency','Frequency','Monetary'])

    st.write('Selected data has', len(X1.columns), 'features')
    st.write('3 principal components has', np.sum(pca.explained_variance_ratio_), 'total variance explanation')

    selected_data_std_pca = pca.transform(selected_data_std)
    selected_data_std_pca = pd.DataFrame(selected_data_std_pca)
    selected_data_std_pca
    
    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[1], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    plt.scatter(selected_data_std_pca[1], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[1], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired')
    st.pyplot()
    