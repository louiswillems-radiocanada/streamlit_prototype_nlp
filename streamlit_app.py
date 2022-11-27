import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from functionforDownloadButtons import download_button
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Wordcloud
from wordcloud import WordCloud
from PIL import Image

# Plotly
import plotly.express as px

# htbuilder
from htbuilder import div, big, h2, styles
from htbuilder.units import rem

# Analyse de sentiment
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer

# Topic modeling - Thématiques
# from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


st.set_page_config(
    page_title="VoC sentiment analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

st.sidebar.image(
    "https://cdn.freebiesupply.com/logos/large/2x/radio-canada-logo-png-transparent.png",
    width=100,
)


def _max_width_():
    max_width_str = f"max-width: 1500px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()


st.sidebar.title("Outil d'analyse exploratoire des données textuelles")
st.sidebar.markdown("------------------------------------------------------------------------------------")

# st.sidebar.markdown("##Upload reviews data :")

uploaded_file = st.sidebar.file_uploader("Charger les données ici :", type=("csv", "xlsx"),)


# ===============================================================================================================================================
# ====================================================== Importation des données  ===============================================================
# ===============================================================================================================================================


if uploaded_file is not None:
    # file_container = st.expander("Visualiser les données ici")
    df = pd.read_excel(uploaded_file)
    df = df.rename(columns = {'Commentaires':'texte'})
    uploaded_file.seek(0)
    # file_container.write(df)

else:
    st.info(
        f"""
            👆 Téléchargez des exemples de fichiers: [exemple.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv) ou [exemple.xlsx](https://docs.google.com/spreadsheets/d/1HR_6CbKNVEP0YpTzKl3HXvt3tVKz6Jz-/edit?usp=share_link&ouid=102390173815691633450&rtpof=true&sd=true)
        """
    )


    st.stop()


# if filename is not None:
#     # df = pd.read_excel("Verbatimdesabonnement.xlsx")
#     df = pd.read_excel(filename)
#     df = df.rename(columns = {'Commentaires':'texte'})


# Valeurs manquantes
valeurs_manquantes = df["texte"].isnull().sum()
st.write("## Le fichier téléchargé comporte ", df.shape[0]," lignes", "et ", valeurs_manquantes ,"valeurs manquantes")


st.markdown("")
st.markdown("")

#### SENTIMENT ANALYSIS
df_sentiment = df.copy()

# STOPWORDS en francais
STOPWORDS = pd.read_csv("stop_words_french.txt")
stopwords = list(STOPWORDS.a)
stopwords = stopwords

# Spacy 
# spacy_stopwords = fr_stop
df_sentiment['texte_sans_stopwords'] = df_sentiment['texte'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

senti_list = []
for i in df_sentiment["texte_sans_stopwords"]:
    vs = tb(i).sentiment[0]
    if (vs > 0):
        senti_list.append('Positif')
    elif (vs < 0):
        senti_list.append('Négatif')
    else:
        senti_list.append('Neutre')  

df_sentiment["sentiment_prediction"]=senti_list


# PREMIERE 1
COLOR_BLACK = "#262730"
COLOR_GREEN = "#3CA358"
COLOR_RED = "#C8171C"

a, b, c, d = st.columns(4)

def display_dial(title, value, color):
    st.markdown(
        div(
            style=styles(
                text_align="center",
                color=color,
                padding=(rem(0.8), 0, rem(3), 0),
            )
        )(
            h2(style=styles(font_size=rem(0.8), font_weight=600, padding=0))(title),
            big(style=styles(font_size=rem(3), font_weight=800, line_height=1))(
                value
            ),
        ),
        unsafe_allow_html=True,
    )

with a:
    # display_dial("% TEXTES POSITIFS", f"{0.6:.2f}", COLOR_BLACK)
    positif = df_sentiment["sentiment_prediction"].value_counts(normalize=True)["Positif"]
    neutre = df_sentiment["sentiment_prediction"].value_counts(normalize=True)["Neutre"]
    negatif = df_sentiment["sentiment_prediction"].value_counts(normalize=True)["Négatif"]

    display_dial("% POSITIFS", f"{positif*100:.2f}", COLOR_GREEN)

with b:
    display_dial(
        "% NÉGATIFS", f"{neutre*100:.2f}", COLOR_RED
    )

with c:
    display_dial(
        "% NEUTRES", f"{negatif*100:.2f}", COLOR_BLACK
    )
    # display_dial("SUBJECTIVITY", f"{0.3434:.2f}", COLOR_BLACK)


st.markdown("------------------------------------------------------------------------------------")

# ===============================================================================================================================================
# ====================================================== Nuage des points =======================================================================
# ===============================================================================================================================================

col1, col2 = st.columns(2)

with col1:

    comment_words_all = ''
    comment_words_pos = ''
    comment_words_neg = ''

    new_words=("a","c'est", 'c’est', "j'ai", "voulais", "ai", "je", ":", ".", ",", "Non", "Non .", "7\n2", "7", "2.")
    for i in new_words:
        stopwords.append(i)
    
    # All
    for val in df.texte:
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words_all += " ".join(tokens)+" "

    # POSITIFS
    df_pos = df_sentiment[df_sentiment["sentiment_prediction"] == "Positif"]
    for val in df_pos.texte:
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words_pos += " ".join(tokens)+" "

    # NÉGATIFS
    df_neg = df_sentiment[df_sentiment["sentiment_prediction"] == "Négatif"]
    for val in df_neg.texte:
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words_neg += " ".join(tokens)+" "


    wordcloud_all = WordCloud(stopwords=stopwords, background_color="white", max_words=500, colormap='cividis').generate(comment_words_all)
    wordcloud_all.to_file("nuagedesmots_all.png")
    image_all = Image.open('nuagedesmots_all.png')
    st.markdown("")
    st.image(image_all, caption='Tous les mots clés', width=620, )

    # Positifs
    wordcloud_pos = WordCloud(stopwords=stopwords, background_color="white", max_words=200, colormap='Greens').generate(comment_words_pos)
    wordcloud_pos.to_file("nuagedesmots_pos.png")
    image_pos= Image.open('nuagedesmots_pos.png')

    # Négatifs 
    wordcloud_neg = WordCloud(stopwords=stopwords, background_color="white", max_words=200, colormap='Reds').generate(comment_words_pos)
    wordcloud_neg.to_file("nuagedesmots_neg.png")
    image_neg= Image.open('nuagedesmots_neg.png')


with col2:
    st.image(image_pos, caption='Positifs', width=450, )
    st.image(image_neg, caption='Négatifs', width=450, )


st.markdown("")
st.markdown("")


# ===============================================================================================================================================
# ====================================================== Bar Chart (Ngrams) =====================================================================
# ===============================================================================================================================================


with st.form(key="my_form1"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Sélectionner les données",
            ["Toutes les données", "Positifs", "Négatifs"],
            help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "Toutes les données":
            df_keywords = df.copy()

        elif ModelType == "Positifs":
            df_keywords = df_pos.copy()

        else:
            df_keywords = df_neg.copy()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )

        Ngrams = st.slider(
            "Ngram",
            value=1,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Enlever les stop words",
            help="Tick this box to remove stop words from the document (currently English only)", value=True
        )


    
    with c2:
                # STOPWORDS en francais
        STOPWORDS = pd.read_csv("stop_words_french.txt")
        stopwords = list(STOPWORDS.a)
        comment_words = ''
        stopwords = stopwords

        new_words=("a","c'est", 'c’est', "j'ai", "voulais", "ai", "je", ":", ".", ",", "Non", "Non .", "7\n2", "7", "2.")
        for i in new_words:
            stopwords.append(i)
        
        # iterate through the csv file
        for val in df.texte:
        
            # typecaste each val to string
            val = str(val)
        
            # split the value
            tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "

        if StopWordsCheckbox:
            StopWords = stopwords
        else:
            StopWords = None

        def ngrams_top(corpus, ngram_range, n=None):
            ### What this function does: List the top n words in a vocabulary according to occurrence in a text corpus.
            vec = CountVectorizer(ngram_range=ngram_range, stop_words= StopWords).fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
            total_list=words_freq[:n]
            df = pd.DataFrame(total_list, columns=['text', 'count'])
            return df

        ngram_df = ngrams_top(df_keywords["texte"], (Ngrams, Ngrams), n=top_N,)

        fig = px.bar(ngram_df, x="text", y="count", template="plotly_white", labels={"ngram": "Unigram", "count": "Fréquence"}, width=1000, height=500).update_layout(title_text='Top 50 des mots les plus fréquents', title_x=0.5)
        st.plotly_chart(fig)

        submit_button = st.form_submit_button(label="✨ Rafraichir")



# ===============================================================================================================================================
# ====================================================== Topic Modeling (Thématiques) ===========================================================
# ===============================================================================================================================================

st.markdown("")
st.markdown("")

# with st.form(key="my_form2"):
#     with st.spinner("L'analyse des thématiques peut prendre quelques minutes..."):

#         ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
#         with c1:
#             ModelType = st.radio(
#                 "Sélectionner les données",
#                 ["Toutes les données", "Positifs", "Négatifs"],
#                 help="Vous avez la possibilité de sélectionner le jeu de données que vous voulez analyser !",
#             )

#             if ModelType == "Toutes les données":
#                 df_keywords = df.copy()

#             elif ModelType == "Positifs":
#                 df_keywords = df_pos.copy()

#             else:
#                 df_keywords = df_neg.copy()

#             top_N = st.slider(
#                 "Nombre de topics",
#                 min_value=1,
#                 max_value=10,
#                 value=5,
#                 help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
#             )

#         with c2:

#             # Although the stop_words parameter was removed in newer versions, you are still able to remove stopwords by using the CountVectorizer!
#             # Note that the CountVectorizer processes the documents after they are clustered which means that you can use it to clean the documents and optimize your topic representations.
#             vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words= fr_stop)
#             topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True, language="french", nr_topics=top_N)


#             @st.cache(allow_output_mutation=True)
#             def load_model():
#                 vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words= fr_stop)
#                 topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True, language="french", nr_topics=top_N)
#                 return topic_model

#             topic_model = load_model()

#             text = df_keywords['texte'].values.tolist() 

#             topics, probs = topic_model.fit_transform(text)
#             topic_labels = topic_model.generate_topic_labels(nr_words=6, topic_prefix=False, word_length=15, separator=" - ")
#             df_topic_model = topic_model.get_topic_info().head(10)


#             # st.dataframe(dd, 600, 500)  # Same as st.write(df)

#             fig1 = topic_model.visualize_documents(text, topics=list(range(10)), custom_labels=True, height=900,)
#             st.plotly_chart(fig1, use_container_width=True)

#             fig2 = topic_model.visualize_barchart(top_n_topics=10, title = "Score des Thématiques")
#             st.plotly_chart(fig2, use_container_width=True)


#             submit_button = st.form_submit_button(label="✨ Rafraichir")


# ===============================================================================================================================================
# ====================================================== Téléchargement du CSV ==================================================================
# ===============================================================================================================================================


st.markdown("## **🎈 Check & download results **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    # Ajouter le timestamp dans le nom du fichier !!
    CSVButton2 = download_button(df_sentiment, "data.csv", "📥 Téléchargement (.csv)")




# with c2:
#     st.table(df.head(10))








































# from bertopic import BERTopic

# # Remove not nulls
# df_text = df.copy()

# new_words=("a","c'est", 'c’est', "j'ai", "voulais", "ai", "je", "veux", "voir", "est", "ca", "aurai", "jai", "cest","qu")

# for i in new_words:
#     stopwords.append(i)

# # Remove stopwords
# df_text['texte_sans_stopwords'] = df_text['texte'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

# # Series to list
# text = df_text['texte_sans_stopwords'].values.tolist()



# st.write ('What would you like to order?')

# icecream = st.checkbox('Ice cream')
# coffee = st.checkbox('Coffee')
# cola = st.checkbox('Cola')

# if icecream:
#     st.write("Great! Here's some more 🍦")

# if coffee: 
#     st.write("Okay, here's some coffee ☕")

# if cola:
#     st.write("Here you go 🥤")



# with st.spinner("L'analyse des thématiques devrait prendre quelques instants"):

#     ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
#     with c1:

#         top_N = st.slider(
#             "# of results",
#             min_value=1,
#             max_value=30,
#             value=10,
#             help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
#         )

#     with c2:

#         @st.cache(allow_output_mutation=True)
#         def load_model():
#             return BERTopic(verbose=True, language="french", nr_topics=5)
#         topic_model = load_model()

#         topics, probs = topic_model.fit_transform(text)


#         topic_labels = topic_model.generate_topic_labels(nr_words=6, topic_prefix=False, word_length=15, separator=" - ")
#         topic_model.set_topic_labels(topic_labels)
#         topic_model.merge_topics(text, topics_to_merge=[0,10])

#         print(topic_model.get_topic_info().shape)
#         dd= topic_model.get_topic_info().head(20)

#     st.balloons()



#     st.dataframe(dd, 600, 500)  # Same as st.write(df)

#     fig1 = topic_model.visualize_documents(text, topics=list(range(10)), custom_labels=True, height=900)
#     st.plotly_chart(fig1, use_container_width=True)

#     fig2 = topic_model.visualize_barchart(top_n_topics=5)
#     st.plotly_chart(fig2, use_container_width=True)

#     st.form_submit_button('Login')

