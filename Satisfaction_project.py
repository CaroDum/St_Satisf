# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:26:46 2022

@author: carol
"""

import streamlit as st
from joblib import dump, load

from streamlit_option_menu import option_menu

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from nltk.stem.snowball import FrenchStemmer  
from langdetect import detect
from  nltk.tokenize import word_tokenize 
import nltk
nltk.download('punkt')
from  nltk.tokenize import PunktSentenceTokenizer  
#install unidecode
from unidecode import unidecode
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


with st.sidebar:
    choose = option_menu("Menu", ["Le sujet", "Problématique", "Exploration Données", "Exploration MétaDonnées","Nuage de mots", "Bag of Words", "Méthodologie", "Modélisation", "Importantes Features", "Conclusion", "A vous de jouer !","Contact"],
                         icons=['journal', 'wrench adjustable circle', 'clipboard-data','clipboard-plus fill','cloud-check','handbag', 'info-circle','building','reception-4','robot','pencil-square','at'],
                         menu_icon="list", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fadcbd"},
        "icon": {"color": "grey", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#ead1dc"},
        "nav-link-selected": {"background-color": "#ff8100"},
    }
    )

image1 = Image.open('Images/sourire_client.jpg')
image2 = Image.open('Images/customer_satisfaction.jpg')
image3 = Image.open('Images/WordCloud_neg.png')
image4 = Image.open('Images/WordCloud_pos.png')
image5 = Image.open('Images/métadonnées.PNG')
image6 = Image.open('Images/df_info.PNG')


df = pd.read_csv('Données/data_satis.csv')
occurences = pd.read_csv('Données/occurences.csv', names = ['mot','occurence'])
data_test =  pd.read_csv('Données/data_test.csv')



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
    st.image(image2)

    title_container1 = st.container()
    col1, col2 = st.columns([1, 1.2])
    with title_container1:
        with col1:
            st.markdown('<h2 style="color: black;">La problématique</h2>', unsafe_allow_html=True)
        with col2:
            st.image(image1, width=100)


    st.write('Il peut être intéressant d’évaluer la satisfaction client pour :')
    st.write('● Étude de qualité sur la Supply Chain : problème de conception, livraison, prix non adapté, durabilité… ')
    st.write('● Étudier si le produit/service correspond bien à l’attente du marché. ')
    st.write('● Synthétiser les feedback, améliorations des clients.  ')
    st.write('● Aider à la réponse ou à la redirection des clients insatisfaits...  ')
    st.write('Pour de nombreux produits/services, la satisfaction des clients se mesure sur les commentaires, avis de sites dédiés (Trustpilot, site distributeur, twitter…).')
    st.write('Il est long et fastidieux mais important de lire et analyser les verbatim qui sont essentiels à la compréhension de la satisfaction client, mais en l’absence d’outils qui permettent de synthétiser ces avis, les procédés sont généralement par échantillonnage.')
    st.write('L’objectif de ce projet est donc d’extraire de l’information de commentaires.  ')
    st.write('Comment prédire la satisfaction d’un client ? ')
    st.write('A travers le commentaire qu’il nous a laissé, nous allons essayer de prédire sa satisfaction globale (Content / pas content), et tenter de prédire le nombre d’étoiles qu’il va donner à l’enseigne.')
    st.write('**--> Nous allons donc mettre en place une modélisation à travers une classification et une régression.** ')





elif choose == "Exploration Données":
    st.image(image2)
    st.markdown('<h2 style="color: black;">Notre DataSet</h2>', unsafe_allow_html=True)

    st.write('**Notre jeu de données de base**')
    
    with st.echo():
        df_base = pd.read_csv("https://assets-datascientest.s3.eu-west-1.amazonaws.com/datasets/reviews_trust.csv")

    st.write('\n ')
    
    @st.cache
    def convert_df(df):
          # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    
    csv1 = convert_df(df_base)  
    st.download_button(
        label="Télécharger le fichier (csv)",
        data=csv1,
        file_name='data_satis.csv',
        mime='text/csv',
        key=1,
        )
    
    st.write('\n ')  

    if st.checkbox('Afficher le jeu de données de base :'):
        st.dataframe(df_base)
        st.write('\n ')
        
        
        
        
    if st.checkbox('Afficher les valeurs manquantes du jdd de base :'):
        st.dataframe(df_base.isna().sum())
    if st.checkbox('Afficher les infos du jeu de données de base :'):
            st.image(image6)
            st.write('Notre base de données est composée de peu de variables : 11 au total.')
            image7 = Image.open('Images/descripvariables_base.PNG')
            st.image(image7)
            st.write("Nous remarquons avec les infos ci-dessus que nous avons énormément de valeurs manquantes pour les variables : reponse / ville /maj / date_commande / ecart.")
            st.write("Nous pouvons déjà supposer que toutes ne seront pas utiles pour l’analyse et la modélisation : en effet, notre objectif est de prédire la note à l’aide du commentaire : nous pouvons donc exclure la variable réponse par exemple qui a été ajoutée après l’obtention de la note (puisque réponse au commentaire")
            st.write("Idem pour les variables client et ville : elles n’apporteront rien dans l’analyse et en plus, il y a beaucoup trop de manquants")

    st.write('\n ')
    st.write('\n ')
    st.write('**Notre jeu de données retravaillé et nettoyé**')
    with st.echo():
        df = pd.read_csv("Données/data_satis.csv")
        
        
    csv2 = convert_df(df_base)  
    st.download_button(
        label="Télécharger le fichier (csv)",
        data=csv2,
        file_name='data_satis.csv',
        mime='text/csv',
        key=2,
        )
    st.write('\n ') 
    
    if st.checkbox('Afficher le jeu de données transformé :'):
        st.dataframe(df)
    if st.checkbox('Afficher les valeurs manquantes du jeu de données transformé :'):
        st.dataframe(df.isna().sum())
    if st.checkbox('Afficher les infos du jeu de données transformé :'):
        st.dataframe(df.describe().T)
        
        
    st.write('\n ')
    st.write("Nous avons dû opérer des modifications dans nos données : en effet, certains commentaires n'étaient pas en français et nous aurions faussé notre modélisation en les laissant")
    st.write("De plus, certaines variables étaient inutiles : elles contenaient trop de valeurs manquantes : nous les avons enlevées")
    st.write('Enfin, nous avons créé des métadonnées pour nous aider dans notre modélisation')
    st.write('\n ')
    st.image(image5, caption="Les métadonnées créées")
    st.write('\n ')
    

    st.markdown('<h6 style="color: black;">La variable STAR</h6>', unsafe_allow_html=True)
    st.write('\n')

    
    if st.checkbox('Afficher les graphes relatifs aux valeurs de star :'):
        image9 = Image.open('Images/graphes_star.png')
        st.image(image9)
        
        st.info('La variable **star**, qui représente la note, est composée de 5 modalités, allant de 1 à 5, 1 représentant la note des clients les plus satisfaits.\n Notons une majorité de notes à 5 (près de 6000 observations), mais suivie par la modalité "1" (la plus basse note) à 5000 observations. \n Notre "ventre mou" est composé des notes moyennes (2 et 3), la note "4" est presque à 4000 observations. \n → on peut en déduire que les notes sont soit bonnes voire très bonnes, soit vraiment très mauvaises, les notes "2" et "3" étant plutôt minoritaires.' )     
        st.write('\n ')

    if st.checkbox('Afficher les stats descriptives sur star :'):
        image10 = Image.open("Images/stats_star.JPG")
        st.image(image10)
        st.info('Le calcul de la moyenne nous confirme une note plutôt élevée à **3.23** / 5.  \n  Et nous constatons que plus 50% des observations ont une note de 4 ou 5 → ce qui correspond à des clients plutôt contents !')

    st.write('\n ')
    st.write('\n ')

    st.markdown('<h6 style="color: black;">La variable COMPANY</h6>', unsafe_allow_html=True)
    st.write('\n ')  
    st.write('Comment est composée la variable **company**, qui représente la ou les enseignes concernées par les avis')
    st.write('\n ')  
    if st.checkbox("Afficher l'analyse sur la variable compagny :"):
        image11 = Image.open("Images/graphes_company.png")
        st.image(image11)
        st.info("Nous n'avons que 2 marques dans le fichier : Veepee et ShowRoomPrivé : 2 spécialistes de la vente événementielle en ligne.  \n   Pour rappel, il s’agit de ventes éphémères sur des produits de grandes marques, avec des réductions importantes pour le client.  \n  Nous pouvons donc comparer ces 2 acteurs, qui œuvrent dans le même domaine : vente en ligne, aucun magasin, envoi par la poste ou relais.  \n  Notons que VeePee est sous-représenté par rapport à ShowRoomPrivé avec moins de 3.000 observations pour plus de 14.000 pour ShowRoomPrivé. (non significatif pour la suite car nous allons modéliser le fichier dans son ensemble)")  
    st.write('\n ')  
    st.write('\n ')  


    st.markdown('<h6 style="color: black;">La variable SOURCE</h6>', unsafe_allow_html=True)
    st.write('\n ')  
    st.write("Comment est composée la variable **Source**, qui représente le ou les sites d'avis déposés par les clients")
    st.write('\n ')  
    if st.checkbox("Afficher l'analyse sur la variable source :"):
        image12 = Image.open("Images/graphes_source.png")
        st.image(image12)
        st.info("Notre base de données est constituée d'avis clients provenant de 2 sites :\n - TrustedShop \n - TrustPilot \n Notons la différence entre les 2 sites d'avis : Trusted shop : les avis clients sont vérifiés, ils font suite à une commande client et pour TrustPilot : ce sont des avis d'internautes, et donc pas forcément vérifiés.")     
    st.write('\n ')  
    st.write('\n ')  


    st.markdown('<h6 style="color: black;">La variable SOURCE vs COMPANY</h6>', unsafe_allow_html=True)
    st.write('\n ')  
    st.write("Quelle est la répartition de la **source** par rapport à l'enseigne ?")
    st.write('\n ')  
    if st.checkbox("Afficher l'analyse :", key =1):
        image13 = Image.open('Images/sourcerepart.JPG')
        st.image(image13)
        st.info("Aucun commentaire pour VeePee sur le site TrustedShop : ces commentaires proviennent exclusivement du site TrustPilot.  \n  Les commentaires pour ShowRoomPrivé SRP proviennent en grande majorité du site TrustedShop **(83%)** ")     
    st.write('\n ')  
    st.write('\n ')  
    
    
    st.markdown('<h6 style="color: black;">La variable STAR vs COMPANY</h6>', unsafe_allow_html=True)
    st.write('\n ')  
    st.write('Quelle est la répartition de la variable **star** par rapport à nos 2 enseignes ?')
    st.write('\n ')  
    if st.checkbox("Afficher l'analyse :", key =2):
        image15 = Image.open("Images/proportion_note_enseigne.png")
        st.image(image15)
        st.info("Énorme différence de notation entre les 2 marques ! Veepee n'a quasiment que des notes = 1, tandis que la distribution pour ShowRoomPrivé suit la distribution générale vu tout à l'heure (logique au vu du nombre d’observations liées à ShowRoom également !)")  
    st.write('\n ')  
    st.write('\n ')  
    st.write('Quelles sont les stats de la variable **star** par rapport à nos 2 enseignes ?')
    st.write('\n ')  
    if st.checkbox("Afficher l'analyse :", key =3):
        image16 = Image.open("Images/star_parenseigne.JPG")
        st.image(image16)
        st.info("Confirmation de ce que le graphique précédent nous indiquait : la moyenne pour VeePee est à **1.46 / 5** pour 3.000 observations, vs **3.61/ 5** pour près de 14.000 observations pour ShowRoomPrivé.  \n  --> Les clients VeePee sont clairement beaucoup plus insatisfaits que ceux de ShowRoom !")
    st.write('\n ')  
    st.write('\n ')  

    st.markdown('<h6 style="color: black;">La variable REPONSE</h6>', unsafe_allow_html=True)
    st.write('\n ')  
    st.write('Quelle enseigne répond le plus au client ?')
    st.write('\n ')  
    if st.checkbox("Afficher l'analyse sur la réponse apportée:"):
        image17 = Image.open("Images/image17.png")
        st.image(image17)
        st.success("**Variable très intéressante, car elle permet de voir si un client mécontent reçoit une réponse de la part de son enseigne. Autrement dit, est-ce que le client reçoit une réponse ? quel est le client a qui on répond (mécontent?) ?**")
        st.info("Instructif car nous avons vu précédemment que VeePee avait une majorité de note à 1 --> en tant qu’enseigne, il serait intéressant de répondre à au client et de pouvoir constater ainsi :  \n  d’une part son mécontentement et tenter de le résoudre, d’autre part de voir si un problème récurrent n'apparaît pas dans les commentaires. \n  Or, dans notre fichier, nous constatons que VeePee ne répond jamais aux clients. ShowRoomPrivé le fait dans près d'un cas sur 2. ")  
        st.error("**Remarque** : cette variable ne doit pas être prise en compte pour expliquer la note car elle intervient forcément après le dépôt de la note et du commentaire sur le site.  \n  Elle peut en tout cas expliquer une non fidélisation des clients. ")
    
    st.write('\n ')  
    st.write('A qui apporte-t-on une réponse ?')
    st.write('\n ')  
    if st.checkbox("Afficher le graphe"):
        image18 = Image.open("Images/star_et_answer.png")
        st.image(image18)
        st.info("Les réponses ne sont pas apportées en fonction de la note. Ce qui est étonnant car il serait productif de répondre aux clients les plus insatisfaits.  \n  De plus en plus de clients déposent des avis sur une multitude de sites, et malgré toutes les technologies pour rassembler ces avis sur une seule et même plateforme permettant de répondre plus facilement au client, il est extrêmement long et coûteux pour les enseignes de répondre au commentaire, de façon personnalisée pour tous les clients.  \n  → il est judicieux dans ce cas de 'choisir ses combats' et de privilégier dans un 1e temps les clients mécontents pour éviter leur départ et leur non-rachat par la suite")
    st.write('\n ')  
    st.write('\n ')  
 
 
############################################################################################
   

elif choose == "Exploration MétaDonnées":
    st.image(image2)
    st.markdown('<h2 style="color: black;">MéTADONNéES</h2>', unsafe_allow_html=True)
    st.write('\n ') 
    st.write("Objectif de cette partie : nous conforter dans nos idées sur le lien entre note et 'aspect' du commentaire. Pour cela, nous avons créé des “métadonnées” --> de nouvelles variables créées à partir du commentaire pour nous aider dans l’analyse.") 
    st.write('Rappel :')
    st.write('\n ')
    st.image(image5, caption="Les métadonnées créées")
    st.write('\n ') 

    st.markdown('<h6 style="color: black;">La métadonnée LONGUEUR DU COMMENTAIRE</h6>', unsafe_allow_html=True)
    st.write('\n ')  
    if st.checkbox("en général, par enseigne et selon les notes données", key=1):
        image19 = Image.open("Images/long_com_describe.JPG")
        image20 = Image.open("Images/long_com_par_ens.JPG")
        image21 = Image.open("Images/long_com_par_star.JPG")
        
        title_container = st.container()
        col1, col2 = st.columns([1, 1.3])
        with title_container:
            with col1:
                st.image(image19)
            with col2:
                st.image(image20)
        
 

        st.info("Si on regarde la longueur du commentaire, nous n’apprenons pas grand-chose. On peut se dire qu’ils sont de longueur variable, avec des petits et de très longs commentaires.  \n  Si on compare les longueurs des commentaires entre les enseignes, cela devient intéressant --> les commentaires chez VeePee sont nettement plus longs que ceux chez ShowRoom : serait-ce le signe d’un client plus mécontent ?")
        st.write('\n ') 

        st.image(image21)
        st.write('\n ') 
        st.info("Clairement, les commentaires des clients les plus mécontents sont plus longs, et au plus le client est content, au plus son commentaire est court ! ")
    st.write('\n ') 
    st.write('\n ') 
    st.markdown('<h6 style="color: black;">La métadonnée MOT NEGATIF</h6>', unsafe_allow_html=True)
    st.write('\n ') 
    st.success('**Pour cette donnée, nous avons construit une liste de mots dits "négatifs" (ex : nul, déteste, jamais, voleur...) et nous avons regardé ensuite si dans notre commentaire, se trouvaient ces mots.  \n  (méthode utilisée pour organiser le flux des mails dans certaines enseignes et en accélérer le traitement**')
    st.write('\n ')  

    if st.checkbox("en général, par enseigne et selon les notes données", key=2):
        image22 = Image.open("Images/mots_neg_describe.JPG")
        image23 = Image.open("Images/mots_neg_par_ens.JPG")
        image24 = Image.open("Images/mots_neg_par_star.JPG")
        
        title_container = st.container()
        col1, col2 = st.columns([1, 1.3])
        with title_container:
            with col1:
                st.image(image22)
            with col2:
                st.image(image23)
        
        
        st.info("Logiquement, nous voyons que plus la note est basse, plus le nombre de mots négatifs est élevé.  De même, quand on regarde la différence entre les 2 enseignes, on voit qu'en moyenne, le nombre de mots négatifs est plus élevé chez VeePee que chez ShowRoom. ")
        st.image(image24)
        st.info("Quand on regarde la différenciation sur note et enseigne, le nombre de mots négatifs diminue bien en fonction de l'augmentation de la note, et ce quelle que soit l'enseigne !")
    st.write('\n ') 
    st.write('\n ') 

    st.markdown('<h6 style="color: black;">La métadonnée POINT EXCL / INTERR</h6>', unsafe_allow_html=True)
    st.write('\n ') 
    if st.checkbox("en général, par enseigne et selon les notes données", key=3):
        st.write('\n ') 
        image25 = Image.open("Images/image25.JPG") #nb pts describe
        image26 = Image.open("Images/image26.JPG") #tc nb_pt / star
        ptsparstar = Image.open("Images/ptsparstar.JPG") #nb pts par star
        ptsparens = Image.open("Images/ptspar_ens.JPG") # nb pts par enseigne

        st.image(image25)
        st.image(image26)

        st.info("En moyenne, une 'zone de point' par commentaire ... mais ça monte jusque 64 !  \n  Le tableau croisé nous indique que pour les données les plus élevées de cette variable, la note est toujours à 1.  \n  En revanche, quand il n'y a pas de zone de ponctuation détectée, la note est bonne ou très bonne (4 ou 5) dans près de 2 cas sur 3.")
        st.write('\n')
        st.write('\n')
        title_container = st.container()
        col1, col2 = st.columns([1, 1.3])
        with title_container:
            with col1:
                st.image(ptsparstar)
            with col2:
                st.image(ptsparens)
        st.info("Comme pour la variable “mot_neg”, la tendance est similaire pour la variable 'nb_pt'' : en moyenne, nous avons plus de 'zone de points' chez VeePee que chez ShowRoom.  \n  Et le nombre de points diminue au fur et à mesure que la note monte, quel que soit l'enseigne considérée.")
   

#################################################################################

elif choose == "Nuage de mots":
    st.image(image2)
    st.markdown('<h2 style="color: black;">Nuage de mots</h2>', unsafe_allow_html=True)
    st.write('\n ')
    st.write('Dans cette partie, nous allons pouvoir visualiser les mots les plus utilisés dans les commentaires, selon que le client soit plutôt content ou mécontent')
    st.write('Les clients **"contents"** sont les clients qui ont mis la note de **4 ou de 5**')
    st.write('Les clients **"mécontents"** sont ceux qui ont noté **entre 1 et 3** inclus')
    st.write('\n ')

        
    if st.checkbox('Afficher le nuage de mots "clients contents" :'):
        st.image(image4, caption="Nuage de Mots 'Clients Contents'")
        st.info("Il y a **67576** mots dans l'ensemble des commentaires clients 'contents")

        st.write('\n ')
    if st.checkbox('Afficher le nuage de mots "clients mécontents" :'):
        st.image(image3, caption="Nuage de Mots 'Clients Mécontents'")
        st.info("Il y a **273905** mots dans l'ensemble des commentaires clients 'mécontents") 
    st.write('\n ')
    st.write('\n ')     
    
    st.write("Vous avez maintenant la possibilité de visualiser le nuage de mots liés aux différentes typologies de clients, c'est à dire, voir les mots utilisés par les clients, en fonction de la note qu'ils ont donnés")
    st.write('\n ')     

    nuage = st.selectbox(
     'Quel nuage souhaitez vous visualiser ?',
     ('les mots liés à la note 1', 'les mots liés à la note 2', 'les mots liés à la note 3','les mots liés à la note 4','les mots liés à la note 5'))
    if nuage == 'les mots liés à la note 1':
        nuage1 = Image.open("Images/nuage1.png")
        st.image(nuage1)
    if nuage == 'les mots liés à la note 2':
        nuage2 = Image.open("Images/nuage2.png")
        st.image(nuage2)
    if nuage == 'les mots liés à la note 3':
         nuage3 = Image.open("Images/nuage3.png")
         st.image(nuage3)
    if nuage == 'les mots liés à la note 4':
        nuage4 = Image.open("Images/nuage4.png")
        st.image(nuage4)
    if nuage == 'les mots liés à la note 5':
        nuage5 = Image.open("Images/nuage5.png")
        st.image(nuage5)         
        
        
        
        

  
    
  
###############################################################################        
    
elif choose == "Bag of Words":
    st.image(image2)
    st.markdown('<h2 style="color: black;">Bag of Words</h2>', unsafe_allow_html=True)
    st.write('\n ')
    st.write("Voir les mots utilisés par les clients et les dénombrer : voilà l'objectif de cette partie")
    st.write('\n ')
    st.write('\n ')
    if st.checkbox('Afficher le tableau entier des occurences :'):
        st.dataframe(occurences)
   
    
    nb = st.slider('Choix du nombre de mots que vous voulez afficher :', 1, 150, 25)
    st.write('\n ')
    st.write('Voici le tableau des ',nb,'mots les plus utilisés par les clients, avec leur occurence :')
    occ =  occurences.sort_values(by = ['occurence'], ascending = False).head(nb)
    st.dataframe(occ)
   
    
   
########################################################################



elif choose == "Méthodologie":
    st.image(image2)
    st.markdown('<h2 style="color: black;">Méthodologie</h2>', unsafe_allow_html=True)
    st.write('\n')
    st.write('Les commentaires des clients laissés après la réception de la commande va permettre au site de faire parler de lui : c’est la **e-réputation**.')
    st.write('Mieux et moins chers qu’une campagne publicitaire ou que des offres envoyées par ciblage client, les commentaires et avis client sont essentiels aux commerces (en ligne et /ou en physique). D’ailleurs, 88% des individus consultent des avis de consommateurs, des forums ou des blogs (dont 44 % « souvent »)')
    st.write('voir : "L’impact de l’e-réputation sur le processus d’achat - IFOP : https://www.ifop.com/publication/limpact-de-le-reputation-sur-le-processus-dachat/ ')
    st.write('\n')
               
    st.write('\n')
    
    st.write('Notre but est donc d’arriver à prédire la satisfaction de notre client en fonction du commentaire laissé sur les sites d’avis. ')
    st.write('--> ainsi, sur des sites autres que avis de consommateurs, nous pourrons prédire la satisfaction de nos clients, et prédire par exemple, quel client sera le + fidèle (intégration d’un programme d’ambassadeurs de la marque) ou prédire les clients “churners” pour lui adresser une offre ou une communication particulière ! ')
    st.write(' La modélisation consiste à appliquer des algorithmes de classification : un client appartient soit à une classe 1 (le client est satisfait) soit à une classe 0 (client non satisfait). ')
    st.write('\n')
    st.write('Le processus de la modélisation se décompose en quatre étapes :')
    image100 = Image.open("Images/etapes_modelisa.JPG")
    st.image(image100)
    st.write('\n')

    st.write('Nous avons choisi pour commencer des modèles de classification “classiques” : **Régression Logistique**, **Decision Tree Classifier** et **Gradient Boosting Classifier**, que nous allons utiliser avec et sans nos métadonnées créées spécialement.  \n  Ensuite, nous nous pencherons sur une tentative de modélisation linéaire mais en nous basant sur la variable Star directement, avec en modèles : **Régression Linéaire**, **Decision Tree Regressor** et **Gradient Boosting Regressor**, toujours avec et sans métadonnées.')
    st.write('\n')

    st.markdown('<h5 style="color: black;">Préparation des données</h5>', unsafe_allow_html=True)
    st.write('\n')
    st.write("___1. Nettoyage du fichier :___  \n  But : Enlever les variables inutiles / les valeurs manquantes, afin de ne pas alourdir le fichier et ainsi alléger le temps de traitement. Nous ne gardons que les variables utiles pour la modélisation : ")
    image101 = Image.open("Images/image101.JPG")
    st.image(image101)
    st.write('Et nous éliminons également les lignes où il n’y a pas de commentaires.')
    st.write('\n')
    st.write('\n')

    
    st.write("___2. Création des séries pour les modèles :___")
    st.write("Pour la classification binaire, nous créons 3 séries :  \n  - une Série **Xmeta** contenant le Commentaire et les métadonnées.  \n  - une Série **Xcom** contenant uniquement la colonne 'Commentaire'.  \n  - une Série y contenant la colonne **target**.")
    st.write('Pour la classification MultiTarget (régression linéaire), nous créons également 3 séries :  \n  - une Série **Xmeta** contenant le Commentaire et les métadonnées.  \n  - une Série **Xcom** contenant uniquement la colonne "Commentaire".  \n  - une Série y contenant la colonne **star** ')
    st.write('_(dans un souci de clarté, nous avons créé des tableaux différents pour chaque type de modélisation_')
    st.write('\n')
    st.write('\n')

    
    st.write("___3. Création des jeux de données de test et d'entraînement___")
    st.write("A partir de nos séries, nous créons des jeux d'entraînement et de test, avec une taille de jeu de test correspondant à **20%** des données au total.  \n  _Toujours dans le souci de clarté et pour ne pas surcharger et trop utiliser le même jeu d'entraînement, nous créons différents jeux de train / test pour les appliquer à nos différents modèles à tester._")
    st.write('\n')
    st.write('\n')
    st.write("___4. Création de nos Pipelines___")
    st.write("Pour notre modèle avec métadonnées, nous avions un problème sur la vectorisation : Il fallait vectoriser notre commentaire, puis ensuite le “raccrocher” à nos métadonnées   \n  → risque d’erreur et difficultés accrues ! ")
    st.write("Nous avons mis en place une Pipeline, qui va nous permettre de traiter nos données textuelles et de modéliser avec nos métadonnées, beaucoup plus facilement. Cette pipeline va traiter en parallèle la vectorisation du commentaire, la normalisation des autres données, enfin, le lancement de la modélisation.  \n  _La Pipeline sera sur 2 “steps” : 'vectorizer' & 'scaler' / 'model'_")    
    st.write('\n')
    st.write('\n')
    st.write('Maintenant : place à la  modélisation :sunglasses:!')
  
    
  ########################################################################


elif choose == "Modélisation":
      st.image(image2)
      st.markdown('<h2 style="color: black;">Modélisation</h2>', unsafe_allow_html=True)
      st.write('\n')
      st.write('__Vous trouverez ci-dessous les résultats de la modélisation de nos modèles testés__')
      st.write('\n')
      
      type_mod = st.radio(
     "Quelle type de modélisation voulez-vous tester ?",
     ('Régression', 'Classification'))
      if type_mod == 'Régression' :
          méta = st.radio(
         "le modèle contient-il des métadonnées ?",
         ('Sans métadonnées', 'Avec métadonnées'))
          if méta == 'Sans métadonnées':
              mod = st.radio(
             "Quel modèle voulez-vous tester ?",
             ('Régression linéaire', 'Decision Tree Regressor','Gradient Boosting Regressor'),index=1)
              if mod == 'Régression linéaire':
                  image220 = Image.open("Images/Modelisation/image220.JPG") #reg lin ss m
                  image221 = Image.open("Images/Modelisation/image221.JPG") #cm reg lin ss m
                  image227 = Image.open("Images/image227.jpg")  
     
                  title_container = st.container()
                  col1, col2 = st.columns([1, 10])
                  with title_container:
                      with col1:
                          st.image(image227,width = 50)
                          st.write('\n')
                          with col2:
                              st.write('_Pour ce modèle, les “prédictions” ont été reclassées sur la bonne échelle pour les comparaisons. (c’est à dire  : toutes les sorties inférieures à 1 ont été reclassées dans le “1”,  toutes les sorties > à 5 ont été reclassées dans le “5”_')
                  st.write('\n')
                  st.write('\n')
                  if st.checkbox('Afficher le rapport de classification :', key = 7):
                      st.image(image220)
                  if st.checkbox('Afficher la matrice de confusion :', key = 7):
                      st.image(image221)
              st.write('\n ')
              if mod == 'Decision Tree Regressor':
                  image222 = Image.open("Images/Modelisation/image222.JPG") #DTR ss m
                  image223 = Image.open("Images/Modelisation/image223.JPG") #cm DTR ss m

                  if st.checkbox('Afficher le rapport de classification :', key = 8):
                      st.image(image222)
                  if st.checkbox('Afficher la matrice de confusion :', key = 8):
                      st.image(image223)
                  st.write('\n ')
                  st.write('\n ')    

              if mod == 'Gradient Boosting Regressor':
                  image224 = Image.open("Images/Modelisation/image224.JPG") #GBR ss m
                  image225 = Image.open("Images/Modelisation/image225.JPG") #cm GBR ss m

                  if st.checkbox('Afficher le rapport de classification :', key = 9):
                      st.image(image224)
                  if st.checkbox('Afficher la matrice de confusion :', key =9):
                      st.image(image225)
          if méta == 'Avec métadonnées':
              mod = st.radio(
             "Quel modèle voulez-vous tester ?",
             ('Régression linéaire', 'Decision Tree Regressor','Gradient Boosting Regressor'),index=1)
              if mod == 'Régression linéaire':
                  image230 = Image.open("Images/Modelisation/image230.JPG") #reg lin av m
                  image231 = Image.open("Images/Modelisation/image231.JPG") #cm reg lin av m
                  image227 = Image.open("Images/image227.jpg")
                  title_container = st.container()
                  col1, col2 = st.columns([1, 10])
                  with title_container:
                      with col1:
                          st.image(image227,width = 50)
                          st.write('\n')
                      with col2:
                          st.write('_Pour ce modèle, les “prédictions” ont été reclassées sur la bonne échelle pour les comparaisons. (c’est à dire  : toutes les sorties inférieures à 1 ont été reclassées dans le “1”,  toutes les sorties > à 5 ont été reclassées dans le “5”_')
                  st.write('\n')
                 
                  if st.checkbox('Afficher le rapport de classification :', key = 10):
                      st.image(image230)
                  if st.checkbox('Afficher la matrice de confusion :', key = 10):
                      st.image(image231)

              if mod == 'Gradient Boosting Regressor':
                  image234 = Image.open("Images/Modelisation/image234.JPG") #GBR av m
                  image235 = Image.open("Images/Modelisation/image235.JPG") #cm GBR av m

                  if st.checkbox('Afficher le rapport de classification :', key = 12):
                      st.image(image234)
                  if st.checkbox('Afficher la matrice de confusion :', key = 12):
                      st.image(image235)
                      
              if mod == 'Decision Tree Regressor':
                  image232 = Image.open("Images/Modelisation/image232.JPG") #DTR av m
                  image233 = Image.open("Images/Modelisation/image233.JPG") #cm DTR av m

                  if st.checkbox('Afficher le rapport de classification :', key = 11):
                      st.image(image232)
                  if st.checkbox('Afficher la matrice de confusion :', key = 11):
                      st.image(image233)

      if type_mod == 'Classification' :
           méta = st.radio(
          "le modèle contient-il des métadonnées ?",
          ('Sans métadonnées', 'Avec métadonnées'))
           if méta == 'Sans métadonnées':
               mod = st.radio(
              "Quel modèle voulez-vous tester ?",
              ('Régression Logistique', 'Decision Tree Classifier','Gradient Boosting Classifier'),index=1)
               if mod == 'Régression Logistique':
                   image200 = Image.open("Images/Modelisation/image200.JPG") #reg log ss m
                   image201 = Image.open("Images/Modelisation/image201.JPG") #cm reg log ss m

                   if st.checkbox('Afficher le rapport de classification :', key = 1):
                       st.image(image200)
                   if st.checkbox('Afficher la matrice de confusion :', key = 1):
                       st.image(image201)
  
               if mod == 'Decision Tree Classifier':
                   image202 = Image.open("Images/Modelisation/image202.JPG") #DTC ss m
                   image203 = Image.open("Images/Modelisation/image203.JPG") #cm DTC ss m

                   if st.checkbox('Afficher le rapport de classification :', key = 2):
                       st.image(image202)
                   if st.checkbox('Afficher la matrice de confusion :', key = 2):
                       st.image(image203)
 
               if mod == 'Gradient Boosting Classifier':
                   image204 = Image.open("Images/Modelisation/image204.JPG") #GBC ss m
                   image205 = Image.open("Images/Modelisation/image205.JPG") #cm GBC ss m

                   if st.checkbox('Afficher le rapport de classification :', key = 3):
                       st.image(image204)
                   if st.checkbox('Afficher la matrice de confusion :', key = 3):
                       st.image(image205)

           if méta == 'Avec métadonnées':
               mod = st.radio(
              "Quel modèle voulez-vous tester ?",
              ('Régression Logistique', 'Decision Tree Classifier','Gradient Boosting Classifier'),index=1)
               if mod == 'Régression Logistique':
                  image210 = Image.open("Images/Modelisation/image210.JPG") #reg log av m
                  image211 = Image.open("Images/Modelisation/image211.JPG") #cm reg log av m

                  if st.checkbox('Afficher le rapport de classification :', key = 4):
                      st.image(image210)
                  if st.checkbox('Afficher la matrice de confusion :', key = 4):
                      st.image(image211)
                      
               if mod == 'Decision Tree Classifier':
                   image212 = Image.open("Images/Modelisation/image212.JPG") #DTC av m
                   image213 = Image.open("Images/Modelisation/image213.JPG") #cm DTC av m

                   if st.checkbox('Afficher le rapport de classification :', key = 2):
                       st.image(image212)
                   if st.checkbox('Afficher la matrice de confusion :', key = 2):
                       st.image(image213)

               if mod == 'Gradient Boosting Classifier':
                   image214 = Image.open("Images/Modelisation/image214.JPG") #GBC av m
                   image215 = Image.open("Images/Modelisation/image215.JPG") #cm GBC av m

                   if st.checkbox('Afficher le rapport de classification :', key = 3):
                       st.image(image214)
                   if st.checkbox('Afficher la matrice de confusion :', key = 3):
                       st.image(image215)
      
        
      
      st.write('\n ')
      st.write('\n ')
      st.write('\n ')

      st.write("__Récap, analyse et commentaires__")
      col1, col2 = st.columns(2)
      
      with col1:
          st.write("____")
          st.write("___Classification___ ")

          if st.checkbox('Afficher tableau récap + commentaires sur les 3 modèles de classification sans métadonnées :', key = 1):
              image206 = Image.open("Images/Modelisation/image206.JPG")  # resum_class_ssm
              st.image(image206)
              st.info('Nous avons déjà de très bons résultats à partir de ces 3 modèles qui en plus n’utilisent pas les métadonnées créées.  \n  Le meilleur est le Reg Log, suivi par le Gradient Boosting.  \n  Le modèle Decision Tree est le moins performant.')
              st.write('\n ')  
            
          if st.checkbox('Afficher tableau récap + commentaires sur les 3 modèles de classification avec métadonnées :', key = 2):
              image216 = Image.open("Images/Modelisation/image216.JPG")  # resum_class_avm
              st.image(image216)
              st.info('Le seul modèle qui s’améliore avec les métadonnées est le Gradient Boosting. Le Log Reg reste très haut, mais avec un F1 Score et un score d’accuracy en très légère baisse.')
              st.write('\n ')  
    
      with col2:
          st.write("____")
          st.write("___Prédiction de score : régression___")

      
          if st.checkbox('Afficher tableau récap + commentaires sur les 3 modèles de régression (prédiction de score) sans métadonnées :', key = 1):
              image226 = Image.open("Images/Modelisation/image226.JPG")  # resum_class_ssm
              st.image(image226)
              st.info('Nous notons des scores très faibles par rapport à ce qu’on a pu obtenir sur la classification. Le meilleur modèle ici est le **Decision Tree** , mais avec une Accuracy à seulement **0.56 !**  \n   Le Lin Reg a un score très moyen : on remarque qu’il classe assez bien les notes extrêmes : **le 1 à 64%** et **le 5 à 70%**, mais le reste n’est pas reconnu. (mais peut- être est-ce dû à notre “reclassement” ?)  \n  Pour le GradientBoostingReg, le comportement pour le classement du 1 est similaire au hasard, et pour le reste c’est encore moins bon : on pourrait presque en déduire que si ce modèle prédit un 2, alors c’est tout sauf un 2 qui va sortir !')

          st.write('\n ')  
            
          if st.checkbox('Afficher tableau récap + commentaires sur les 3 modèles de régression (prédiction de score) avec métadonnées :', key = 2):
              image236 = Image.open("Images/Modelisation/image236.JPG")  # resum_class_avm
              st.image(image236)
              st.info('Aucune amélioration pour le Lin Reg.  \n  Le Gradient Boosting s’améliore très (trop) légèrement avec l’ajout des métadonnées.  \  Pas d’amélioration ni détérioration pour le Decision Tree Regressor.  \n  En résumé : les modèles ne sont pas meilleurs avec les métadonnés')
              st.write('\n ')  
              st.write('\n ')  
          

   
########################################################################


elif choose == "Modélisation2":
    st.image(image2)
    st.markdown('<h2 style="color: black;">Modélisation</h2>', unsafe_allow_html=True)


    st.markdown('<h5 style="color: black;">Modèle de classification sans métadonnées</h5>', unsafe_allow_html=True)
    st.write('\n')
    
    st.write("___1. Logistic Regression sans métadonnées___ ")
    image200 = Image.open("Images/Modelisation/image200.JPG") #reg log ss m
    image201 = Image.open("Images/Modelisation/image201.JPG") #cm reg log ss m

    if st.checkbox('Afficher le rapport de classification :', key = 1):
        st.image(image200)
    if st.checkbox('Afficher la matrice de confusion :', key = 1):
        st.image(image201)
    st.write('\n ')
    st.write('\n ')


    st.write("___2. Decision Tree Classifier sans métadonnées___ ")
    image202 = Image.open("Images/Modelisation/image202.JPG") #DTC ss m
    image203 = Image.open("Images/Modelisation/image203.JPG") #cm DTC ss m

    if st.checkbox('Afficher le rapport de classification :', key = 2):
        st.image(image202)
    if st.checkbox('Afficher la matrice de confusion :', key = 2):
        st.image(image203)
    st.write('\n ')
    st.write('\n ')    
   
    
    st.write("___3. Gradient Boosting Classifier sans métadonnées___ ")
    image204 = Image.open("Images/Modelisation/image204.JPG") #GBC ss m
    image205 = Image.open("Images/Modelisation/image205.JPG") #cm GBC ss m

    if st.checkbox('Afficher le rapport de classification :', key = 3):
        st.image(image204)
    if st.checkbox('Afficher la matrice de confusion :', key = 3):
        st.image(image205)
    st.write('\n ')
    st.write('\n ')    
    
    
    if st.checkbox('Afficher tableau récap + commentaires sur ces 3 modèles :', key = 1):
        image206 = Image.open("Images/Modelisation/image206.JPG")  # resum_class_ssm
        st.image(image206)
        st.info('Nous avons déjà de très bons résultats à partir de ces 3 modèles qui en plus n’utilisent pas les métadonnées créées.  \n  Le meilleur est le Reg Log, suivi par le Gradient Boosting.  \n  Le modèle Decision Tree est le moins performant.')

    st.write('\n ')
    st.write('\n ')  
    
## ----------------------------------------------##
   
    st.markdown('<h5 style="color: black;">Modèle de classification avec métadonnées</h5>', unsafe_allow_html=True)
    st.write('\n')
    
    st.write("___1. Logistic Regression avec métadonnées___ ")
    image210 = Image.open("Images/Modelisation/image210.JPG") #reg log av m
    image211 = Image.open("Images/Modelisation/image211.JPG") #cm reg log av m

    if st.checkbox('Afficher le rapport de classification :', key = 4):
        st.image(image210)
    if st.checkbox('Afficher la matrice de confusion :', key = 4):
        st.image(image211)
    st.write('\n ')
    st.write('\n ')


    st.write("___2. Decision Tree Classifier avec métadonnées___ ")
    image212 = Image.open("Images/Modelisation/image212.JPG") #DTC av m
    image213 = Image.open("Images/Modelisation/image213.JPG") #cm DTC av m

    if st.checkbox('Afficher le rapport de classification :', key = 5):
        st.image(image212)
    if st.checkbox('Afficher la matrice de confusion :', key = 5):
        st.image(image213)
    st.write('\n ')
    st.write('\n ')    
   
    
    st.write("___3. Gradient Boosting Classifier sans métadonnées___ ")
    image214 = Image.open("Images/Modelisation/image214.JPG") #GBC av m
    image215 = Image.open("Images/Modelisation/image215.JPG") #cm GBC av m

    if st.checkbox('Afficher le rapport de classification :', key = 6):
        st.image(image214)
    if st.checkbox('Afficher la matrice de confusion :', key =6):
        st.image(image215)
    st.write('\n ')
    st.write('\n ')    
    
    
    if st.checkbox('Afficher tableau récap + commentaires sur ces 3 modèles :', key = 2):
        image216 = Image.open("Images/Modelisation/image216.JPG")  # resum_class_avm
        st.image(image216)
        st.info('Le seul modèle qui s’améliore avec les métadonnées est le Gradient Boosting. Le Log Reg reste très haut, mais avec un F1 Score et un score d’accuracy en très légère baisse.')

        
### --------------------------------------------------------------------- ###

    st.markdown('<h5 style="color: black;">Modèle de régression : prédiction de score sans métadonnées</h5>', unsafe_allow_html=True)
    st.write('\n')
    st.success('**Remarque : Pour effectuer ce type de modélisation, toutes les “prédictions” de ses modèles ont dû être arrondies à l’entier près pour pouvoir établir la comparaison.**')
    st.write('\n')
    
    st.write("___1. Linear Regression sans métadonnées___ ")
    image220 = Image.open("Images/Modelisation/image220.JPG") #reg lin ss m
    image221 = Image.open("Images/Modelisation/image221.JPG") #cm reg lin ss m
    image227 = Image.open("Images/image227.jpg")
    
    
    title_container = st.container()
    col1, col2 = st.columns([1, 10])
    with title_container:
        with col1:
            st.image(image227,width = 50)
            st.write('\n')
        with col2:
            st.write('_Pour ce modèle, les “prédictions” ont été reclassées sur la bonne échelle pour les comparaisons. (c’est à dire  : toutes les sorties inférieures à 1 ont été reclassées dans le “1”,  toutes les sorties > à 5 ont été reclassées dans le “5”_')
    st.write('\n')
    
    
    st.write('\n')
    if st.checkbox('Afficher le rapport de classification :', key = 7):
        st.image(image220)
    if st.checkbox('Afficher la matrice de confusion :', key = 7):
        st.image(image221)
    st.write('\n ')
    st.write('\n ')


    st.write("___2. Decision Tree Classifier sans métadonnées___ ")
    image222 = Image.open("Images/Modelisation/image222.JPG") #DTR ss m
    image223 = Image.open("Images/Modelisation/image223.JPG") #cm DTR ss m

    if st.checkbox('Afficher le rapport de classification :', key = 8):
        st.image(image222)
    if st.checkbox('Afficher la matrice de confusion :', key = 8):
        st.image(image223)
    st.write('\n ')
    st.write('\n ')    
   
    
    st.write("___3. Gradient Boosting Regressor sans métadonnées___ ")
    image224 = Image.open("Images/Modelisation/image224.JPG") #GBR ss m
    image225 = Image.open("Images/Modelisation/image225.JPG") #cm GBR ss m

    if st.checkbox('Afficher le rapport de classification :', key = 9):
        st.image(image224)
    if st.checkbox('Afficher la matrice de confusion :', key =9):
        st.image(image225)
    st.write('\n ')
    st.write('\n ')    
    
    
    if st.checkbox('Afficher tableau récap + commentaires sur ces 3 modèles :', key = 3):
        image226 = Image.open("Images/Modelisation/image226.JPG")  # resum_class_ssm
        st.image(image226)
        st.info('Nous notons des scores très faibles par rapport à ce qu’on a pu obtenir sur la classification. Le meilleur modèle ici est le **Decision Tree** , mais avec une Accuracy à seulement **0.56 !**  \n   Le Lin Reg a un score très moyen : on remarque qu’il classe assez bien les notes extrêmes : **le 1 à 64%** et **le 5 à 70%**, mais le reste n’est pas reconnu. (mais peut- être est-ce dû à notre “reclassement” ?)  \n  Pour le GradientBoostingReg, le comportement pour le classement du 1 est similaire au hasard, et pour le reste c’est encore moins bon : on pourrait presque en déduire que si ce modèle prédit un 2, alors c’est tout sauf un 2 qui va sortir !')

    st.write('\n ')
    st.write('\n ')  

## ----------------------------------------------##
   
    st.markdown('<h5 style="color: black;">Modèle de régression : prédiction de score avec métadonnées</h5>', unsafe_allow_html=True)
    st.write('\n')
    st.success('**Remarque : Pour effectuer ce type de modélisation, toutes les “prédictions” de ses modèles ont dû être arrondies à l’entier près pour pouvoir établir la comparaison.**')
    st.write('\n')

    st.write("___1. Linear Regression avec métadonnées___ ")
    
    image230 = Image.open("Images/Modelisation/image230.JPG") #reg lin av m
    image231 = Image.open("Images/Modelisation/image231.JPG") #cm reg lin av m

    title_container = st.container()
    col1, col2 = st.columns([1, 10])
    with title_container:
        with col1:
            st.image(image227,width = 50)
            st.write('\n')
        with col2:
            st.write('_Pour ce modèle, les “prédictions” ont été reclassées sur la bonne échelle pour les comparaisons. (c’est à dire  : toutes les sorties inférieures à 1 ont été reclassées dans le “1”,  toutes les sorties > à 5 ont été reclassées dans le “5”_')
    st.write('\n')
   
    if st.checkbox('Afficher le rapport de classification :', key = 10):
        st.image(image230)
    if st.checkbox('Afficher la matrice de confusion :', key = 10):
        st.image(image231)
    st.write('\n ')
    st.write('\n ')


    st.write("___2. Decision Tree Regressor avec métadonnées___ ")
    image232 = Image.open("Images/Modelisation/image232.JPG") #DTR av m
    image233 = Image.open("Images/Modelisation/image233.JPG") #cm DTR av m

    if st.checkbox('Afficher le rapport de classification :', key = 11):
        st.image(image232)
    if st.checkbox('Afficher la matrice de confusion :', key = 11):
        st.image(image233)
    st.write('\n ')
    st.write('\n ')    
   
    
    st.write("___3. Gradient Boosting Regressor avec métadonnées___ ")
    image234 = Image.open("Images/Modelisation/image234.JPG") #GBR av m
    image235 = Image.open("Images/Modelisation/image235.JPG") #cm GBR av m

    if st.checkbox('Afficher le rapport de classification :', key = 12):
        st.image(image234)
    if st.checkbox('Afficher la matrice de confusion :', key = 12):
        st.image(image235)
    st.write('\n ')
    st.write('\n ')    
    
    
    if st.checkbox('Afficher tableau récap + commentaires sur ces 3 modèles :', key = 4):
        image236 = Image.open("Images/Modelisation/image236.JPG")  # resum_class_avm
        st.image(image236)
        st.info('Aucune amélioration pour le Lin Reg.  \n  Le Gradient Boosting s’améliore très (trop) légèrement avec l’ajout des métadonnées.  \  Pas d’amélioration ni détérioration pour le Decision Tree Regressor.  \n  En résumé : les modèles ne sont pas meilleurs avec les métadonnés')

    



    # les tableaux de résultats pour chaque modèle et analyse
    # résumé et conclusion 







elif choose == "A vous de jouer !":
    st.image(image2)
    st.markdown('<h2 style="color: black;">A vous de tester le modèle </h2>', unsafe_allow_html=True)





    # mettre une zone de texte pour entrer un commentaire
    txt = ""
    txt = st.text_area("Entrez votre commentaire, et nous allons voir si le modèle prédit correctement votre sentiment :", value = "Entrez ici votre commentaire")
     
    
    # faire passer à la fonction TOUTPROPRE ce comm
    stemmer = FrenchStemmer()  
    # fonction stemming qui retrouve la racine pour chaque mot de mots, une liste (de mots) passée en paramètre.
    # on va l'appliquer ensuite à notre liste de mots noirs, puis à notre commentaire
    def stemming(liste): 
        sortie = [] 
        stemmer = FrenchStemmer() 
        for l in liste: 
            rac = stemmer.stem(l) 
            if rac not in sortie: 
                sortie.append(rac) 
                return sortie  


### création de notre liste de mots négatifs #
# liste non exhaustive !#

    mots_neg = ['annul', 'deçu', 'decevant', 'catastrophe','casse','mecontent','misere', 'galere',
              'fraude','avocat','arnaque','argent','remboursement','non','nul','manquant',
              'retard','desastre','erreur','errone','vol','vide','abime','perdu','delai',
              'amertune', 'abattement', 'absurdite', 'accablement', 'affliction', 'angoisse', 'assombrissement',
              'austerite', 'agitation', 'agressivite', 'aigreur', 'animosite', 'anxiete', 'agacement',
              'cher', 'couter', 'crainte', 'chagrin', 'colere', 'consternation', 'courroux', 'critique', 
              'contradiction', 'crainte', 'cruaute',
              'deranger', 'dependre', 'desoler', 'difficile', 'decouragement', 'degout', 'depression', 
              'desabusement', 'desenchantement','desesperance', 'desolation', 'douleur', 'depit', 'doute', 'desarme',
              'ennui', 'eplorement', 'ebullition', 'effervescence', 'emportement', 'etroitesse','exasperation', 
              'excitation', 'excede',  'embetement', 'embarras', 'etourdi', 'enervement',
              'froideur', 'folie', 'facherie', 'fulminant', 'fureur', 'frustration', 'frousse', 'fermete', 'faiblesse',
              'grossièrete',
              'haine', 'hargne', 'honte', 'humiliation', 'hautain',
              'impossible', 'inconscience', 'indelicatesse', 'indifference', 'insensibilite',  
              'inquietude', 'impatience', 'indignation', 'irascibilite', 'irritabilite', 'irrationalite', 'impuissant',
              'long', 'laideur', 'lassitude', 'lourdeur',
              'mal', 'malheureusement',  'mais', 'mauvais','mauvaise', 'marre', 'machiavelisme', 'maladresse', 'malaise',
              'malheur','maussaderie', 'mecontentement', 'mefiance','melancolie','moche', 'mochete', 'monotonie',
              'morosite', 'moquerie', 'mechancete', 'menti',
              'neanmoins', 'negatif', 'navrement', 'niais', 'nigaud', 'noirceur',
              'orgueil','pauvrete', 'peine', 'platitude', 'petit', 'peu', 'peur', 'presumer', 'prejuger' ,'pretention', 'problème',
              'rage', 'ressentiment', 'rogne', 'rudesse', 'resignation',
              'sagacite', 'severite', 'sombreur', 'souci', 'souffrance', 'stupidite', 
              'susceptibilité', 'superiorite', 'sottise','tristesse', 'vengeance', 'violence', 'vulgarite'
              'voleur','voleurs','escroc','reclamation', 'extremement','deception','decu','deçu','bizarre','facture',
              'incomplete','fuir','deconseiller','deception','fuyez','desoeuvrant','deplorable','abominable', 'cauchemar']

    # on passe notre liste dans la fonction stemming pour n'avoir que les racines des mots négatifs
    # ne passe pas sur StreamLit
    # mots_neg = stemming(mots_neg)

    
    # fonction qui va permettre de compter le nb de mots négatifs contenus dans le commentaire 
    
    def mot_noir(texte):
        mot = word_tokenize(texte.lower(), language = 'french')
        nb = 0
        for m in mot:
            #m = stemmer.stem(m)
            if m in mots_neg :
                nb += 1
        return nb
    
    
    stop_words = set(stopwords.words('french'))
    stop_words.update(["?", "!", ".", ",", ":", ";", "-", "--", "...","€"]) 
    stop_words.update(['\"'])  
    stop_words.update(["\'"])
    stop_words.update(['"'])  
    stop_words.update(["'"])
    stop_words.update(["\’"])
    stop_words.update(["’"])
    stop_words.update(['1','2','3','4','5','6','7','8','9','0'])
    stop_words.update(['a', "j'ai", 'si',"n'ai",'ça','ca','cela', "n'est",])
    stop_words.update(['donc', "c'est", 'plus',"tout",'très',"fois",'rien', 'ni','jour','là', "qu'il", 'fait'])  
    stop_words.update(['avoir', "quand", 'comme',"faire",'car','alors','chez','suite','après'])  
    stop_words.update(['cet', "cette", 'leurs',"leur", 'meme', 'apres','etait'])  
    stop_words.update(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])  

    # importation du tokenizer
    from nltk.tokenize import TweetTokenizer  
    tokenizer = TweetTokenizer()


    # définition de la fct pour enlever les mots "vides" du commentaire
    # garantie une meilleure perf de la vectorisation
    # on doit découper notre texte en mot (sinon ça fait lettre par lettre)
    # puis renvoyer un str (ici parole) pour que la vectorisation puisse se faire 
    # on ajoute un espace entre les mots sinon cestpastrèslisible

    def stop_words_filtering(texte) :
        # on coupe en mot la phrase que l'on veut analyser
        phrase = word_tokenize(texte.lower(), language = 'french')
        tokens = []
        for mot in phrase:
            if mot not in stop_words:
                parole = ""
                tokens.append(mot)
                for t in tokens:
                    parole += str(t)
                    parole += ' '
        return parole
    
    
    
    def toutpropre(x):
    ## 1e partie : création des métadonnées:
      long_com = len(str(x))  # longueur du commentaire

      # nb points exclamation / interrogation
      pt_exc = re.compile(r"\?+")
      pt_int = re.compile(r"!+")
      interr = len(pt_int.findall(str(x)))
      exclam = len(pt_exc.findall(str(x)))
      points = interr + exclam
  
      # suite majuscule : on conserve la variable nb_maj car plus liée à la note
      maj = re.compile(r"[A-Z]")
      majmot = re.compile(r"[A-Z]{2,}")
  
      nb_maj = len(maj.findall(str(x)))
      mot_maj = len(majmot.findall(str(x)))

 
     
      neg = mot_noir(x)   # compte le nb de mots négatifs dans le commentaire

      # 2nde partie : on nettoie notre commentaire : enlève ponctuation , maj, mots vide:
      x_OK = stop_words_filtering(x)    
      x_OK = x_OK.lower()      # minuscule
      x_OK = unidecode(x_OK)    # On enlève les accents
      x_OK = re.sub("[^a-zA-Z]", " ", x_OK)   # On enlève la ponctuation
      x_OK = re.sub('\s+',' ',x_OK)        # On enlève les espaces en trop

      # 3e partie : on va créer un DF avec ces métadonnées et le commentaire nettoyé :
 
      df_x = pd.DataFrame(list(zip([long_com],[neg],[interr], [exclam], [points] , [mot_maj],[nb_maj],[x_OK] )), 
                      columns = ['long_com', 'mots_neg', 'trouve_int', 'trouve_exc','nb_pt', 'mot_maj', 'nb_maj','com_OK'])
      return df_x
  

  # Chargement du modèle (à faire sur l'app Streamlit)
    pipeLR = load("Modeles/pipeLR.joblib") 
    pipeDT = load("\Projet DS\Modeles/pipeDT.joblib") 
    pipeGBC = load('Modeles/pipeGBC.joblib') 

    pipeLRM = load('Modeles/pipeLRM.joblib') 
    pipeDTM = load('Modeles/pipeDTM.joblib') 
    pipeGBCM = load('Modeles/pipeGBCM.joblib') 

    pipeRL = load('Modeles/pipeRL.joblib') 
    pipeDTR = load('Modeles/pipeDTR.joblib') 
    pipeGBR = load('Modeles/pipeGBR.joblib') 

    pipeRLm = load('Modeles/pipeRLm.joblib') 
    pipeDTRm = load('Modeles/pipeDTRm.joblib') 
    pipeGBRm = load('Modeles/pipeGBRm.joblib') 
       
   
    # faire choisir le modèle pour prédiction
    option = st.selectbox(
     'Quel modèle voulez-vous tester ?',
     ('Régression Logisitique', 
      'Decision Tree Classifier', 
      'Gradient Boosting Classifier',
      'Régression Linéaire',
      'Decision Tree Regressor',
      'Gradient Boosting Regressor'))
    st.write('\n')
    

    radio = st.radio("Avec ou sans métadonnées ?", ('Avec Métadonnées', 'Sans Métadonnées'))
    


    st.write('Vous avez choisi :', option, radio)
    st.write('\n')
    
    if option == 'Régression Logisitique' and radio == 'Sans Métadonnées':
        mod = pipeLR
    if option == 'Decision Tree Classifier' and radio == 'Sans Métadonnées':
        mod = pipeDT
    if option == 'Gradient Boosting Classifier' and radio == 'Sans Métadonnées':
        mod = pipeGBC

    if option == 'Régression Logisitique' and radio == 'Avec Métadonnées':
        mod = pipeLRM
    if option == 'Decision Tree Classifier' and radio == 'Avec Métadonnées':
        mod = pipeDTM
    if option == 'Gradient Boosting Classifier' and radio == 'Avec Métadonnées':
        mod = pipeGBCM
        
        
    if option == 'Régression Linéaire' and radio == 'Sans Métadonnées':
        mod = pipeRL
    if option == 'Decision Tree Regressor' and radio == 'Sans Métadonnées':
        mod = pipeDTR
    if option == 'Gradient Boosting Regressor' and radio == 'Sans Métadonnées':
        mod = pipeGBR

    if option == 'Régression Linéaire' and radio == 'Avec Métadonnées':
        mod = pipeRLm
    if option == 'Decision Tree Regressor' and radio == 'Avec Métadonnées':
        mod = pipeDTRm
    if option == 'Gradient Boosting Regressor' and radio == 'Avec Métadonnées':
        mod = pipeGBRm
        
        
        
         
    # donner la prédiction pour la CLASSIFICATION
    if option == 'Régression Logisitique' or option == 'Decision Tree Classifier' or option == 'Gradient Boosting Classifier':
        if txt == "Entrez ici votre commentaire":
            resultc = "non"
        else:
            resultc = mod.predict(toutpropre(txt))[0]

    
        if resultc == 0:
            st.error("--> Client **mécontent** : la note prédite mauvaise : **1, 2 ou 3**     :disappointed:")
        if resultc == 1:
            st.success("  -->    Client **satisfait**, la note prédite sera élevée : **4 ou 5**      :sunglasses:  ")
        if resultc == "non":
                pass
        
        
    # donner la prédiction pour la REGRESSION
    if option == 'Régression Linéaire' or option == 'Decision Tree Regressor' or option == 'Gradient Boosting Regressor':
        if txt == "Entrez ici votre commentaire":
            resultr = "non"
        else:
            resultr = mod.predict(toutpropre(txt))[0]

        resultr = np.round(resultr,0)
        if resultr <= 0:
            resultr = 1
            if resultr > 5:
                resultr =5
        
        if resultr == 1:
            st.error("   -->     Client **très mécontent** : la note prédite est **1**     😖")
        if resultr == 2:
            st.error("   -->     Client **mécontent** : la note prédite est **2**     :🙁")
        if resultr == 3:
            st.info("   -->     Client **mitigé** : la note prédite est **3**      😑")
        if resultr == 4:
            st.success("   -->     Client **satisfait** : la note prédite est **4**      🙂")
        if resultr == 5:
             st.success("  -->    Client **très satisfait**, la note prédite est **5**      🥰 ")
        if resultr == "non":
             pass
        
        
#################################################################################    

elif choose == "Importantes Features":
    st.image(image2)   
    st.markdown('<h2 style="color: black;">Importantes Features</h2>', unsafe_allow_html=True)
    st.write('\n ')   
    st.write('\n ')   
    st.write('Après des essais de visualisation sur le Bag of Word, nous voudrions voir quels sont les mots qui contribuent le plus à la construction du modèle. Nous utilisons ici le **Logistic Regression avec métadonnées.**')
    if st.checkbox('Afficher Top 40 des mots avec contribution Positive'):
        image270 = Image.open("Images/image270.png")
        st.image(image270) 
    st.write('\n')     
    st.write('\n')    
    if st.checkbox('Afficher Top 40 des mots avec contribution Négative'):
        image271 = Image.open("Images/image271.png")
        st.image(image271)  
    st.write('\n')
    st.write('\n')    
    
    
    if st.checkbox('Afficher Top 40 des mots les plus contributeurs, en positif comme en négatif'):
        image260 = Image.open("Images/image260.JPG")
        st.image(image260) 
        st.write('\n')
        st.info("Les mots les plus contributeurs en positif sont les mots **“satisfait”**, **“parfait”** et **“rapide”**.  \n  Les mots les plus contributeurs côté négatif sont **”mois”**, **“arnaque”**, et **“mauvaise”**.  \n  Le mot “mois” est étonnant... Mais quand on lit quelques commentaires, on s'aperçoit que certains clients ont attendu très longtemps leur colis")
        st.write('\n')
    st.write('\n')    
    st.write('\n')    
    if st.checkbox('Afficher Importantes Features pour le modèle Gradient Boosting avec métadonnées'):
        image261 = Image.open("Images/image261.JPG")
        image262 = Image.open("Images/image262.JPG")

        title_container = st.container()
        col1, col2 = st.columns([1, 1.2])
        with title_container:
            with col1:
                st.image(image261, width = 350)
            with col2:
                st.image(image262, width = 440)
        st.write('\n')
        st.info('La métadonnée "mot _neg" est la plus contributrice.La contribution est quasi-nulle pour les autres métadonnées.')



#################################################################################    
elif choose == "Conclusion":
    st.image(image2)   
    st.markdown('<h2 style="color: black;">Conclusion / Perspectives</h2>', unsafe_allow_html=True)
    st.write('\n ')   
    st.write('\n ')   
    
    
    if st.checkbox('Récapitulatif et choix du meilleur modèle testé :'):
        st.write('En résumé, nous retenons la modélisation par classification : elle est efficace et donne de bons résultats.  \n  Et dans ce type de modèle en particulier, la régression Logistique sans métadonnées semble la plus fiable et la plus sûre : **la prédiction est sûre à 90%** !')
        st.write('\n ')
        st.write('La régression linéaire est moins adaptée pour ce type de données : nous aurions sûrement dû rééquilibrer nos groupes avant de modéliser, ou augmenter la taille de nos données, afin de pouvoir avoir des groupes plus équilibrés.')
        st.write('\n ')
        st.write("Enfin, nous remarquons que quelque soit le modèle considérée, l’ajout de métadonnées n’est pas synonyme d’amélioration de la prédiction : soit nous devons améliorer l’élaboration de nos métadonnées (en choisir d’autres, augmenter leur nombre, ne choisir que certaines qui seraient les plus contributrices…) soit nous pouvons simplement nous fier à la vectorisation de notre commentaire pour obtenir le plus fiablement notre prédiction ! ")
    st.write('\n ')
    st.write('\n ')    

    if st.checkbox('Prédictions sur un fichier "neuf" :'):
        st.write('Nous avons mis en place une “mini-base” de commentaires fictifs, et avons tenté une prédiction par quelques-uns des modèles vus plus haut. ')
        image250 = Image.open("Images/image250.JPG")
        image251 = Image.open("Images/image251.JPG")
        st.write('\n ')
        st.image(image250, width = 950) 
        st.image(image251, width = 450) 
        st.info('On pourrait penser que Gradient Boosting Classifier fonctionne avec la métadonnée “long_com” : c’est à dire qu’au plus le commentaire est long, au plus il reconnaît le côté négatif du commentaire. Sauf qu’avec ou sans cette métadonnée, il fonctionne de la même façon. Si on exclut ces commentaires longs (8e , 11e, 21e), il passe à côté de ces prédictions.  \n  → Je penche plutôt vers les mots dans le commentaire et donc la matrice de vectorisation : plus elle est riche, au plus le Gradient Boosting Classifier est performant. ')
        
    st.write('\n ')
    st.write('\n ')  

    if st.checkbox('Pour aller plus loin :'):
        st.write('Nous avons réussi à prédire la note donnée à partir du commentaire du client.  \n  Et maintenant ?')
        st.write('1. En application directe, nous pourrions aller tester notre modèle sur d’autres sites, pour d’autres enseignes   \n  → En extrayant les commentaires de site qui ne sont pas forcément des des sites de “notes” (ex TrustPilot), nous pourrions savoir à partir du commentaire du client la façon dont il nous note et pouvoir établir une vision globale de l’enseigne à travers l’ensemble des canaux d’expression des clients : FaceBook / Insta / Avis Vérifiés / Messenger / Pages Jaunes …')
        st.write('En fonction de cette note prédite, nous pouvons établir un plan de réponses aux clients les plus mécontents   \n  → il est **indispensable** de répondre aux clients et en priorité à ceux les plus mécontents, sous peine qu’ils partent à la concurrence, et encore pire, qu’ils parlent de leur mauvaise expérience à son entourage.')
        st.write('\n ')
        st.write("**Un client satisfait en parle à 3 autres. Un mécontent, à 11**")
        st.write('\n')
        st.write("3. Toujours en fonction de la note prédite, et en continuant notre analyse du texte : il serait très intéressant de pouvoir creuser sur la raison du mécontentement du client   \n  → pour trouver les 3 sujets sur lesquels travailler :  \n  - établir un constat “pourquoi ça ne marche pas”  \n   - simplifier le process “moi, en tant que client, je souhaite …”  \n  - établir un plan d’actions “3 mesures pro-client”  \n  - mesurer et # soit pérenniser le plan d’action   # soit réadapter le plan d’actions ")
        st.write('\n')
        st.write('\n')
        
        title_container = st.container()
        col1, col2 = st.columns([1, 2.1])
        with title_container:
            with col2:
                st.image(image1, width = 200)

    st.write('\n ')
    st.write('\n ')  



#################################################################################
elif choose == "Contact":
    st.image(image2)   
    st.markdown('<h2 style="color: black;">Contact</h2>', unsafe_allow_html=True)
    
    
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Calibri'; color: #ff8100;} 
    </style> """, unsafe_allow_html=True)
    
    st.markdown('<p class="font">Qui suis-je ? </p>', unsafe_allow_html=True)
    
    
    title_container0 = st.container()
    col1, col2 = st.columns([1, 5])
    imagecd = Image.open("Images/photoCDU.PNG")
    with title_container0:
        with col1:
            st.image(imagecd, width=100)
        with col2:
            st.markdown('<h3 style="color: orange;">Caroline Dumoulin</h3>', unsafe_allow_html=True)
        
    st.write('\n')    
    st.write("Experte en optimisation des process internes et données avec pour objectif **la satisfaction client**​")
    st.write("20 ans d'expérience dont 6 années en tant que responsable Expérience client / Analyse ​Satisfaction Client / Retail​")
    st.write(" Optimisation du traitement des données, analyse des indicateurs et préconisations ​d'actions correctrices.​")
    st.write("Mon projet : **accompagner votre entreprise ​dans le développement de la satisfaction de vos clients**")
  


    st.markdown('<p class="font">Me contacter </p>', unsafe_allow_html=True)
    
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Votre nom') #Collect user feedback
        Email=st.text_input(label='Email') #Collect user feedback
        Message=st.text_input(label='Message') #Collect user feedback
        submitted = st.form_submit_button('Envoyer')
        if submitted:
            st.write('Merci pour votre intérêt. Je vous recontacte dans les 24h :) ')
