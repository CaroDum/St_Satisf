# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:26:46 2022

@author: carol
"""

import streamlit as st
from joblib import dump, load


#from streamlit.script_run_context import get_script_run_ctx
#install streamlit-option-menu

from streamlit_option_menu import option_menu

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib_inline as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report  
#install langdetect
from langdetect import detect
from  nltk.tokenize import word_tokenize 
import nltk
nltk.download('punkt')
from  nltk.tokenize import PunktSentenceTokenizer  
from nltk.stem.snowball import FrenchStemmer  
#install unidecode
from unidecode import unidecode

import re


with st.sidebar:
    choose = option_menu("Menu", ["Le sujet", "Problématique", "Exploration Données", "Nuage de mots", "Modélisation", "A vous de jouer !","Contact"],
                         icons=['house', 'wrench adjustable circle', 'clipboard-data','cloud-check-fill'  , 'building','pencil-square','at'],
                         menu_icon="list", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fadcbd"},
        "icon": {"color": "grey", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#ead1dc"},
        "nav-link-selected": {"background-color": "#ff8100"},
    }
    )

image1 = Image.open('sourire client.jpg')
image2 = Image.open('customer satisfaction.jpg')

# \\Users\carol\OneDrive\Bureau\






if choose == "Le sujet":
    st.title("Projet Customer Satisf'Action")
    st.image(image2, caption="Customer Satisf'Action")
             
    st.markdown('<h2 style="color: black;">Présentation du sujet</h2>', unsafe_allow_html=True)

    st.write('Pourquoi le sujet de la satisfaction client fait autant parler aujourd’hui ?')
    st.markdown('Depuis toujours, le client représente le nerf de la guerre dans les enseignes : **sans client → pas de commerce !**')   
    st.write("Les mentalités changent, la vie également : on ne peut plus se permettre de perdre des clients, la fidélisation des clients est essentielle à la bonne santé de l'entreprise")
    st.write('La guerre des prix fait rage : si on ne peut plus retenir le client uniquement sur une promo ou un prix, il faut aller le retenir avec les services qu’on va lui proposer, ')
    st.write('dans leurs diversités, leurs intérêts et surtout dans leurs qualités ! ')
    st.write('Un service client par ex ouvert 7j/7, c’est bien, mais si le client obtient une réponse lors du 1er contact, c’est vraiment mieux 😊  ')
    st.write('Dès lors se posent plusieurs questions : comment fidéliser nos clients ? Comment savoir que le client est en train de churner ? Comment le retenir ? **Que pense-t-il vraiment ?  Peut-on le prévoir ?**')
    st.write('Voilà le but de ce projet **Satisf’Action !** Car c’est dans *l’action* que qu’il faut se battre pour évoquer et imposer la pensée **“client centric”**')



elif choose == "Problématique":


    title_container1 = st.container()
    col1, col2 = st.columns([1, 1.2])
    with title_container1:
        with col1:
            st.markdown('<h2 style="color: black;">La problématique</h2>', unsafe_allow_html=True)
        with col2:
            st.image(image1, width=100)


    st.write(' il peut être intéressant d’évaluer la satisfaction client pour :')
    st.write('● Étude de qualité sur la Supply Chain : problème de conception, livraison, prix non adapté, durabilité… ')
    st.write('● Étudier si le produit/service correspond bien à l’attente du marché. ')
    st.write('● Synthétiser les feedback, améliorations des clients.  ')
    st.write('● Aider à la réponse ou à la redirection des clients insatisfaits...  ')
    st.write('Pour de nombreux produits/services, la satisfaction des clients se mesure sur les commentaires, avis de sites dédiés (Trustpilot, site distributeur, twitter…).')
    st.write('Il est long et fastidieux mais important de lire et analyser les verbatim qui sont essentiels à la compréhension de la satisfaction client, mais en l’absence d’outils qui permettent de synthétiser ces avis, les procédés sont généralement par échantillonnage.')
    st.write('L’objectif de ce projet est donc d’extraire de l’information de commentaires.  ')
    st.write('Comment prédire la satisfaction d’un client ? ')
    st.write('A travers le commentaire qu’il nous a laissé, nous allons essayer de prédire sa satisfaction globale (Content / pas content), et tenter de prédire le nombre d’étoiles qu’il va donner à l’enseigne.')
    st.write('**--> nous allons donc mettre en place une modélisation à travers une classification et une régression.** ')





elif choose == "Exploration Données":
    st.markdown('<h2 style="color: black;">Notre DataSet</h2>', unsafe_allow_html=True)

    st.write('Notre jeu de données de base ')
    with st.echo():
        df_base = pd.read_csv("https://assets-datascientest.s3.eu-west-1.amazonaws.com/datasets/reviews_trust.csv")

    st.write('\n ')
    st.write('Notre jeu de données retravaillé et nettoyé')
    with st.echo():
        df = pd.read_csv('\\Users\carol\OneDrive\Bureau\data_satis.csv')

    if st.checkbox('Afficher le jeu de données de base :'):
        st.dataframe(df_base)

    if st.checkbox('Afficher les valeurs manquantes du jdd de base :'):
        st.dataframe(df_base.isna().sum())

    if st.checkbox('Afficher le jeu de données transformé :'):
        st.dataframe(df)

    if st.checkbox('Afficher les valeurs manquantes du jeu de données transformé :'):
        st.dataframe(df.isna().sum())
        
        
    st.write('\n ')
    st.write("Nous avons dû opérer des modifications dans nos données : en effet, certains commentaires n'étaient pas en français et nous aurions faussé notre modélisation en les laissant")
    st.write("De plus, certaines variables étaient inutiles : elles contenaient trop de valeurs manquantes : nous les avons enlevées")
    st.write('Enfin, nous avons créé des métadonnées pour nous aider dans notre modélisation')
    st.write('\n ')






#################################################################################

elif choose == "Nuage de mots":
    st.markdown('<h2 style="color: black;">Nuage de mots</h2>', unsafe_allow_html=True)







elif choose == "Modélisation":
    st.markdown('<h2 style="color: black;">Modélisation</h2>', unsafe_allow_html=True)

# Chargement du modèle (à faire sur l'app Streamlit)
    pipeLR = load('pipeLR.joblib') 
    pipeDT = load('pipeDT.joblib') 
    pipeGBC = load('pipeGBC.joblib') 

    pipeLRM = load('pipeLRM.joblib') 
    pipeDTM = load('pipeDTM.joblib') 
    pipeGBCM = load('pipeGBCM.joblib') 

    pipeRL = load('pipeRL.joblib') 
    pipeDTR = load('pipeDTR.joblib') 
    pipeGBR = load('pipeGBR.joblib') 

    pipeRLm = load('pipeRLm.joblib') 
    pipeDTRm = load('pipeDTRm.joblib') 
    pipeGBRm = load('pipeGBRm.joblib') 








elif choose == "A vous de jouer !":
    st.markdown('<h2 style="color: black;">A vous de tester le modèle</h2>', unsafe_allow_html=True)

    
elif choose == "Contact":
    st.header('Contact :', 'Contact')

    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Calibri'; color: #ff8100;} 
    </style> """, unsafe_allow_html=True)
    
    st.markdown('<p class="font">Qui suis-je ? </p>', unsafe_allow_html=True)
    
    
# petite description: mettre un tableau ?
# mettre le lien du CV

    title_container0 = st.container()
    col1, col2 = st.columns([1, 5])
    imagecd = Image.open('photo CDU.PNG')
    with title_container0:
        with col1:
            st.image(imagecd, width=100)
        with col2:
            st.markdown('<h3 style="color: orange;">Caroline Dumoulin</h3>', unsafe_allow_html=True)

#\\Users\carol\OneDrive\Images\

    st.markdown('<p class="font">Me contacter </p>', unsafe_allow_html=True)
    
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Votre nom') #Collect user feedback
        Email=st.text_input(label='Email') #Collect user feedback
        Message=st.text_input(label='Message') #Collect user feedback
        submitted = st.form_submit_button('Envoyer')
        if submitted:
            st.write('Merci pour votre intérêt. Je vous recontacte dans les 24h :) ')
