import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

# Keywords Relevance
from keybert import KeyBERT

# Topic modeling - Thématiques
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


from functionforDownloadButtons import download_button


st.set_page_config(
    page_title="Outil d'analyse exploratoire des données textuelles",
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

uploaded_file = st.sidebar.file_uploader("Cliquer sur le bouton ci-dessous", type=("csv"),)


# ===============================================================================================================================================
# ====================================================== Importation des données  ===============================================================
# ===============================================================================================================================================
## DEV
# df = pd.read_csv("dataset.csv")
# df = df.rename(columns={ df.columns[0]: "texte" })

if uploaded_file is not None:
    # file_container = st.expander("Visualiser les données ici")
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns={ df.columns[0]: "texte" })
    uploaded_file.seek(0)
    # file_container.write(df)
else:
    st.info(
        f"""
            👆 Téléchargez un exemple de fichier: [dataset.csv](https://drive.google.com/file/d/1F7z3EF-fXuI3DgpWdEqn1LXV2AmQXYfo/view?usp=share_link)
        """
    )
    st.stop()


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
    positif = df_sentiment["sentiment_prediction"].value_counts(normalize=True)["Positif"]*100
    neutre = df_sentiment["sentiment_prediction"].value_counts(normalize=True)["Neutre"]*100
    negatif = df_sentiment["sentiment_prediction"].value_counts(normalize=True)["Négatif"]*100

    display_dial("POSITIFS", f"{positif:.0f}%", COLOR_GREEN)

with b:
    display_dial(
        "NÉGATIFS", f"{neutre:.0f}%", COLOR_RED
    )

with c:
    display_dial(
        "NEUTRES", f"{negatif:.0f}%", COLOR_BLACK
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


# # ===============================================================================================================================================
# # ====================================================== Bar Chart (Ngrams) =====================================================================
# # ===============================================================================================================================================


with st.form(key="Form1"):

    a0, a1, a2, a3, a4 = st.columns([0.07, 1, 0.07, 5, 0.07])

    with a1:
        ModelType = st.radio(
            "Sélectionner les données",
            ["Toutes les données", "Positifs", "Négatifs"],
            help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "Toutes les données":
            df_frq = df.copy()

        elif ModelType == "Positifs":
            df_frq = df_pos.copy()

        else:
            df_frq = df_neg.copy()

        top_N = st.slider(
             "Nombre de résultats",
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

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",)

        StopWordsCheckbox = st.checkbox(
            "Enlever les stop words",
            help="Tick this box to remove stop words from the document (currently English only)", value=True
        )

    with a3:
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

        ngram_df = ngrams_top(df_frq["texte"], (Ngrams, Ngrams), n=top_N,)

        fig = px.bar(ngram_df, x="text", y="count", template="plotly_white", labels={"ngram": "Unigram", "count": "Fréquence"}, width=1000, height=500).update_layout(title_text='Mots-clés les plus fréquents',)
        st.plotly_chart(fig)

        submit_button1 = st.form_submit_button(label="✨ Rafraichir")


# # ===============================================================================================================================================
# # ====================================================== Keyword Extractor ======================================================================
# # ===============================================================================================================================================

# st.markdown("")
# st.markdown("")

# with st.form(key="my_form2"):
#     with st.spinner("L'analyse des mots-clés les plus importants peut prendre quelques secondes..."):

#         b0, b1, b2, b3, b4 = st.columns([0.07, 1, 0.07, 5, 0.07])

#         with b1:
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
#                 "Nombre de résultats",
#                 min_value=1,
#                 max_value=50,
#                 value=10,
#                 help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
#             )

#             Ngrams = st.slider(
#                 "Ngram",
#                 value=5,
#                 min_value=1,
#                 max_value=5,
#                 help="""The maximum value for the keyphrase_ngram_range.

#     *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

#     To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",)


#             StopWordsCheckbox = st.checkbox("Enlever les stop words", help="Tick this box to remove stop words from the document (currently English only)", value=False)

#         with b3:
#             @st.experimental_singleton
#             def load_model():
#                 model = KeyBERT("distilbert-base-nli-mean-tokens") #distilbert-base-nli-mean-tokens
#                 return model

#             kw_model = load_model()

#             text = df_keywords['texte'].values.tolist() 
#             doc = ' '.join(map(str, text))
#             # doc = """The 2022 FIFA World Cup is an international association football tournament contested by the men's national teams of FIFA's member associations. The 22nd FIFA World Cup, it is taking place in Qata"""

#             if StopWordsCheckbox:
#                 StopWords = stopwords
#             else:
#                 StopWords = None

#             keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(Ngrams, Ngrams), nr_candidates=30, top_n=top_N, stop_words=StopWords)
            
#             df = (
#                 pd.DataFrame(keywords, columns=["Mots-clés/Phrases ", "Importance"])
#                 .sort_values(by="Importance", ascending=False)
#                 .reset_index(drop=True)
#             )

#             df.index += 1

#             # Add styling
#             cmGreen = sns.light_palette("green", as_cmap=True)
#             cmRed = sns.light_palette("red", as_cmap=True)
#             df = df.style.background_gradient(
#                 cmap=cmGreen,
#                 subset=[
#                     "Importance",
#                 ],
#             ).hide_index()

#             format_dictionary = {
#                 "Importance": "{:.0%}",
#             }

#             df = df.format(format_dictionary)
#             st.table(df)


#             submit_button2 = st.form_submit_button(label="Rafraichir")


# ===============================================================================================================================================
# ====================================================== Topic Modeling (Thématiques) ===========================================================
# ===============================================================================================================================================

st.markdown("")
st.markdown("")


with st.form(key="my_form3"):
    with st.spinner("L'analyse des thématiques peut prendre quelques minutes..."):
        c0, c1, c2, c3, c4 = st.columns([0.07, 1, 0.07, 5, 0.07])


        with c1:
            df_c = df_sentiment.copy()
            df_pos_c = df_sentiment[df_sentiment["sentiment_prediction"] == "Positif"]
            df_neg_c = df_sentiment[df_sentiment["sentiment_prediction"] == "Négatif"]

            ModelType_c = st.radio(
                "Sélectionner les données",
                ["Toutes les données", "Positifs", "Négatifs"],
                help="Vous avez la possibilité de sélectionner le jeu de données que vous voulez analyser !",
            )
            if ModelType_c == "Toutes les données":
                df_topics = df_c.copy()

            elif ModelType_c == "Positifs":
                df_topics = df_pos_c.copy()

            else:
                df_topics = df_neg_c.copy()

            top_N_c = st.slider(
                "Nombre de topics",
                min_value=1,
                max_value=10,
                value=5,
                help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
            )

        with c3:

            # Although the stop_words parameter was removed in newer versions, you are still able to remove stopwords by using the CountVectorizer!
            # Note that the CountVectorizer processes the documents after they are clustered which means that you can use it to clean the documents and optimize your topic representations.

            @st.experimental_singleton
            def load_model_topic_modeling():
                vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words= stopwords)
                topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True, language="french", nr_topics=top_N_c)
                return topic_model

            topic_model = load_model_topic_modeling()

            text = df_topics['texte'].values.tolist() 

            topics, probs = topic_model.fit_transform(text)
            topic_labels = topic_model.generate_topic_labels(nr_words=6, topic_prefix=False, word_length=15, separator=" - ")
            df_topic_model = topic_model.get_topic_info().head(10)


            # st.dataframe(dd, 600, 500)  # Same as st.write(df)

            fig1 = topic_model.visualize_documents(text, topics=list(range(10)), custom_labels=True, height=900,)
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = topic_model.visualize_barchart(top_n_topics=10, title = "Score des Thématiques")
            st.plotly_chart(fig2, use_container_width=True)


            submit_button3 = st.form_submit_button(label="✨ Rafraichir")


# ===============================================================================================================================================
# ====================================================== Téléchargement du CSV ==================================================================
# ===============================================================================================================================================


st.markdown("## **Télécharger les résultats en .csv **")

st.header("")

d0, d1, d2, d3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with d1:
    # Ajouter le timestamp dans le nom du fichier !!
    CSVButton2 = download_button(df_sentiment, "data.csv", "📥 Téléchargement (.csv)")




# with c2:
#     st.table(df.head(10))

