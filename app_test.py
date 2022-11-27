import numpy as np
import pandas as pd
import streamlit as st

# Plotly
import plotly.express as px

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

uploaded_file = st.sidebar.file_uploader("Charger les données ici :", type=("csv", "xlsx"),)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns = {'Commentaires':'texte'})
    uploaded_file.seek(0)

else:
    st.info(
        f"""
            👆 Téléchargez des exemples de fichiers: [exemple.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv) ou [exemple.xlsx](https://docs.google.com/spreadsheets/d/1HR_6CbKNVEP0YpTzKl3HXvt3tVKz6Jz-/edit?usp=share_link&ouid=102390173815691633450&rtpof=true&sd=true)
        """
    )


    st.stop()


valeurs_manquantes = df["texte"].isnull().sum()
st.write("## Le fichier téléchargé comporte ", df.shape[0]," lignes", "et ", valeurs_manquantes ,"valeurs manquantes")

df_sentiment = df.copy()

@st.cache(allow_output_mutation=True)
def load_model():
    vectorizer_model = CountVectorizer(ngram_range=(1, 1))
    topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True, language="french", nr_topics=5)
    return topic_model

topic_model = load_model()

text = df['texte'].values.tolist() 

topics, probs = topic_model.fit_transform(text)
topic_labels = topic_model.generate_topic_labels(nr_words=6, topic_prefix=False, word_length=15, separator=" - ")
df_topic_model = topic_model.get_topic_info().head(10)


fig1 = topic_model.visualize_documents(text, topics=list(range(10)), custom_labels=True, height=900,)
st.plotly_chart(fig1, use_container_width=True)

fig2 = topic_model.visualize_barchart(top_n_topics=10, title = "Score des Thématiques")
st.plotly_chart(fig2, use_container_width=True)











































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

