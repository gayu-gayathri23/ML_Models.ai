import numpy as np
import pandas as pd
import os
import google.generativeai as genai
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)

from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score)


# to get methods from analysis .py

from analysis import suggest_improvements, generate_summary

key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model=genai.GenerativeModel('gemini-2.5-flash-lite')

st.set_page_config(page_title='ML Models Demo', page_icon='🤖', layout='wide')
st.title(':green[ML model Automation] 📊🤖')
st.header('Streamlit App to get CSV and target as input and performs ML algoriths')


uploaded_file=st.file_uploader('Upload your file here 📝',type=['csv'])

if uploaded_file:
    st.markdown('### Preview')
    df=pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    target=st.selectbox(':blue[Select your target 🎯]',df.columns)
    st.write(f":red[Target variable:] {target}")
    
    if target:
        X=df.drop(columns=[target]).copy()
        y=df[target].copy()
        
        #preprocessing
        
        num=X.select_dtypes(include=np.number).columns.tolist()
        cat=X.select_dtypes(exclude=np.number).columns.tolist()
        
        #missing value treatment
        
        X[num]=X[num].fillna(X[num].median())
        X[cat]=X[cat].fillna('Missing data')
        
        #encoding
        X=pd.get_dummies(data=X,columns=cat,drop_first=True,dtype=int)
        
        # for categoric target
        
        
        if y.dtype=='object':
            label= LabelEncoder()
            y=label.fit_transform([y])
            
        # DETECT the PROBLEM TYPE
        
        # detect the problem type
        if df[target].dtype == 'object' or len(np.unique(y)) <=10:
            problem_type = 'Classification'
            
        else:
            problem_type = 'Regression'
            
            
        st.write(f'### PROBLEM TYPE: {problem_type}')
        
        #train test split
        
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #SCALING
        #fit_transform on train data
        #transform on test data
        
        for i in xtrain.columns:
            s=StandardScaler()
            xtrain[i]=s.fit_transform(xtrain[[i]])
            xtest[i]=s.transform(xtest[[i]])
            
        #MODEL BUILDING
        
        results=[]
        
        if problem_type=='Regression':
            models={'Linear Regression': LinearRegression(),
                     'Random Forest Regressor': RandomForestRegressor(random_state=42),
                     'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)}
            
            for name,model in models.items():
                model.fit(xtrain,ytrain)
                ypred=model.predict(xtest)
                
                                
                results.append({'Model':name,
                                'R2 Score':round(r2_score(ytest,ypred), 3),
                                'MSE':round(mean_squared_error(ytest,ypred), 3),
                                'RMSE':round(np.sqrt(mean_squared_error(ytest,ypred)), 3)})                      
        
        else:
            models={'Logistic Regression': LogisticRegression(),
                     'Random Forest Classifier': RandomForestClassifier(random_state=42),
                     'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42)}
            
            for name,model in models.items():
                model.fit(xtrain,ytrain)
                ypred=model.predict(xtest)
                
                results.append({'Model':name,
                                'Accuracy':round(accuracy_score(ytest,ypred), 3),
                                  'Precision':round(precision_score(ytest,ypred, average='weighted'), 3),
                                  'Recall':round(recall_score(ytest,ypred, average='weighted'), 3),
                                  'F1 Score':round(f1_score(ytest,ypred, average='weighted'), 3)})
                
    
        
        
        results_df=pd.DataFrame(results)
        st.write(f'#### :[Results]')
        st.dataframe(results_df)
        
        if problem_type=='Regression':
            st.bar_chart(results_df.set_index('Model')['R2 Score'])
            st.bar_chart(results_df.set_index('Model')['RMSE'])
        else:
            st.bar_chart(results_df.set_index('Model')['Accuracy'])
            st.bar_chart(results_df.set_index('Model')['F1 Score'])
        
        
        
        #AI INSIGHTS
        
        if st.button('Generate Summary'):
            summary=generate_summary(results_df)
            st.write(summary)
            
        if st.button('Suggest Improvements'):
            suggest=generate_summary(results_df)
            st.write(suggest)
            
        
                              