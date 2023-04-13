import streamlit as st
import pandas as pd
import plotly.express as px
import functions
import appy

st.set_page_config(layout = "wide", page_title='Visualize')

st.header("Visualize Data")
st.write('<p style="font-size:160%">You will be able to:</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 1. See the whole data</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 2. Get column names,non null info, data types info</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 3. Get the count of Null values</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 4. Distribution of target using plot</p>', unsafe_allow_html=True)
functions.space()
st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)
file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
dataset = st.file_uploader(label = '')

st.sidebar.header('Import Dataset to Use Available Features:')

if dataset:
    if file_format == 'csv' or use_defo:
        df = pd.read_csv(dataset)
    else:
        df = pd.read_excel(dataset)
    
    st.subheader('Dataframe:')
    n, m = df.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
    st.dataframe(df)


    all_vizuals = ['Info', 'null values',  'Target Analysis']
    functions.sidebar_space(3)         
    vizuals = st.sidebar.multiselect("Choose which visualizations you want to see 👇", all_vizuals)

    if 'Info' in vizuals:
        st.subheader('Info:')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(functions.df_info(df))

    if 'null values' in vizuals:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There are no null values in your dataset.')
        else:
            c1, c2, c3 = st.columns([0.5, 2, 0.5])
            c2.dataframe(functions.df_isnull(df), width=1500)
            functions.space(2)
            
        
    if 'Target Analysis' in vizuals:
        st.subheader("Select target column:")    
        target_column = st.selectbox("", df.columns, index = len(df.columns) - 1)
    
        st.subheader("Histogram of target column")
        fig = px.histogram(df, x = target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)

    num_columns = df.select_dtypes(exclude = 'object').columns
    cat_columns = df.select_dtypes(include = 'object').columns
