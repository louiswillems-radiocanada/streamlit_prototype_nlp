import streamlit as st
import numpy as np
import pandas as pd
from pandas import DataFrame
from keybert import KeyBERT

# For Flair (Keybert)
# from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
# For download buttons
from functionforDownloadButtons import download_button
# import os
# import json



###################################
# from st_aggrid import AgGrid
# from st_aggrid.grid_options_builder import GridOptionsBuilder
# from st_aggrid.shared import JsCode

###################################



st.image(
    "https://site-cbc.radio-canada.ca/media/3166/logo-rc-bg-white-1920x1080.jpg",
    width=250,
)

# st.title("Outil d'exploration de données textuelles")



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

### THEME COLOR
# st.write('Contents of the `config.toml` file of this app')

# st.code("""
# [theme]
# primaryColor="#F39C12"
# backgroundColor="#2E86C1"
# secondaryBackgroundColor="#AED6F1"
# textColor="#FFFFFF"
# font="monospace"
# """)



c29, c30, c31 = st.columns([1, 8, 1])

with c30:
    st.sidebar.header('Paramètres')
    nb_clusters = st.sidebar.slider('Nombre de segments:', 0, 10, 5)
    st.write("Nombre de segments pour l'analyse:", nb_clusters)


with c30:

    # uploaded_file = st.file_uploader(
    #     "",
    #     key="1",
    #     help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    # )

    uploaded_file = st.file_uploader("Upload reviews data:", type=("csv", "xlsx"))

    if uploaded_file is not None:
        file_container = st.expander("Visualiser les données ici")
        df = pd.read_excel(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(df)

    else:
        st.info(
            f"""
                👆 Téléchargez les exemples de fichiers: [exemple.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv) ou [exemple.xlsx](https://docs.google.com/spreadsheets/d/1HR_6CbKNVEP0YpTzKl3HXvt3tVKz6Jz-/edit?usp=share_link&ouid=102390173815691633450&rtpof=true&sd=true)
            """
        )
    

        st.stop()




with st.expander('About this app'):
    st.write('This app shows the various ways on how you can layout your Streamlit app.')
    st.image('https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png', width=250)



# from st_aggrid import GridUpdateMode, DataReturnMode

# gb = GridOptionsBuilder.from_dataframe(df)
# # enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
# gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
# gb.configure_selection(selection_mode="multiple", use_checkbox=True)
# gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
# gridOptions = gb.build()


# with c30:
#     response = AgGrid(
#         shows,
#         gridOptions=gridOptions,
#     )



## Preprocessing
with st.form(key='my_form0'):


    df_texte = df.rename(columns = {'Commentaires':'texte'})

    import collections
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from wordcloud import WordCloud, STOPWORDS
    from PIL import Image

    # STOPWORDS en francais
    STOPWORDS = pd.read_csv("stop_words_french.txt")
    stopwords = list(STOPWORDS.a)
    comment_words = ''
    stopwords = stopwords

    new_words=("a","c'est", 'c’est', "j'ai", "voulais", "ai", "je", ":", ".", ",", "Non", "Non .", "7\n2", "7", "2.")
    for i in new_words:
        stopwords.append(i)
    
    # iterate through the csv file
    for val in df_texte.texte:
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words += " ".join(tokens)+" "


    wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=1000).generate(comment_words)
    wordcloud.to_file("nuagedesmots.png")

    image = Image.open('nuagedesmots.png')



    with c30:
        st.markdown("")
        st.markdown("")
        st.image(image, caption='Nuage des mots', width=500, )

    
    from sklearn.feature_extraction.text import CountVectorizer



    def ngrams_top(corpus, ngram_range, n=None):
        ### What this function does: List the top n words in a vocabulary according to occurrence in a text corpus.
        vec = CountVectorizer(ngram_range=ngram_range, stop_words= stopwords).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        total_list=words_freq[:n]
        df = pd.DataFrame(total_list, columns=['text', 'count'])
        return df

    unigram_df = ngrams_top(df_texte["texte"], (2,2), n=20)

    # fancier interactive plot using plotly express
    import plotly.express as px
    fig = px.bar(unigram_df, x="text", y="count", title="Counts of top unigrams", template="plotly_white", labels={"ngram": "Unigram", "count": "Count"})
    
    with c30:
        st.plotly_chart(fig)



    from bertopic import BERTopic

    # Remove not nulls
    df_text = df_texte.copy()

    new_words=("a","c'est", 'c’est', "j'ai", "voulais", "ai", "je", "veux", "voir", "est", "ca", "aurai", "jai", "cest","qu")

    for i in new_words:
        stopwords.append(i)

    # Remove stopwords
    df_text['clean_text'] = df_text['texte'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

    # Series to list
    text = df_text['clean_text'].values.tolist()


    with c30:
        st.write ('What would you like to order?')

        icecream = st.checkbox('Ice cream')
        coffee = st.checkbox('Coffee')
        cola = st.checkbox('Cola')

        if icecream:
            st.write("Great! Here's some more 🍦")

        if coffee: 
            st.write("Okay, here's some coffee ☕")

        if cola:
            st.write("Here you go 🥤")



    with st.spinner("L'analyse des thématiques devrait prendre quelques instants"):
        ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
        with c1:

            top_N = st.slider(
                "# of results",
                min_value=1,
                max_value=30,
                value=10,
                help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
            )

        with c2:

            @st.cache(allow_output_mutation=True)
            def load_model():
                return BERTopic(verbose=True, language="french", nr_topics=5)
            topic_model = load_model()

            topics, probs = topic_model.fit_transform(text)


            topic_labels = topic_model.generate_topic_labels(nr_words=6, topic_prefix=False, word_length=15, separator=" - ")
            topic_model.set_topic_labels(topic_labels)
            topic_model.merge_topics(text, topics_to_merge=[0,10])

            print(topic_model.get_topic_info().shape)
            dd= topic_model.get_topic_info().head(20)

    st.balloons()


    with c30:
        st.dataframe(dd, 600, 500)  # Same as st.write(df)

    fig1 = topic_model.visualize_documents(text, topics=list(range(10)), custom_labels=True, height=900)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = topic_model.visualize_barchart(top_n_topics=5)
    st.plotly_chart(fig2, use_container_width=True)

    st.form_submit_button('Login')


## Preprocessing














st.markdown("")
st.markdown("## **📜 Paste document **")
with st.form(key="my_form1"):
    
    
    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["DistilBERT (Default)", "Flair"],
            help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "Default (DistilBERT)":
            # kw_model = KeyBERT(model=roberta)

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT(model=roberta)

            kw_model = load_model()

        else:
            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("distilbert-base-nli-mean-tokens")

            kw_model = load_model()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
            # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.
            
Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.

""",
        )

    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )

        MAX_WORDS = 500
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Get me the data!")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()

keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    diversity=Diversity,
)

st.markdown("## **🎈 Check & download results **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")

st.header("")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
