# main.py

#-----------------Libraries------------------------#
import streamlit as st
import pandas as pd
import numpy as np
import streamlit_option_menu as menu
import matplotlib.pyplot as plt
from textblob import TextBlob
import pickle
import sklearn

#-----------------Page Set-up------------------------#
st.set_page_config(
    page_title="FRIENDS: The one with Machine Learning",
    page_icon="",
    #layout=""
)

st.subheader("FRIENDS: The one with Machine Learning")

#-----------------Menu------------------------#
selected_option = menu.option_menu("", options=["Text Classification", "Visualization", "Script"],
                                  icons=["","bi-bar-chart", "bi-search"],
                                  orientation="horizontal")

#-----------------Load Data and Model------------------------#
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Load the data
csv = 'data/friends_data.csv'
data = load_data(csv)

# Load the model
filename = "models/text_classification_model.pkl"
vectorizer_filename = "models/vectorizer.pkl"
loaded_model = pickle.load(open(filename, "rb"))
loaded_vectorizer = pickle.load(open(vectorizer_filename, "rb"))

#-------------Text Classification------------------------#
if selected_option == "Text Classification":
    st.write("Who says the line?")
    dialogue = st.text_input("Write a FRIENDS line here")
    if st.button("Predict"):
        if dialogue == "":
            st.warning("Please write your FRIENDS sample dialog line in the text input.")
            st.warning("e.g., 'We were on a break'")
        else:
            dialogue_transformed =  loaded_vectorizer.transform([dialogue])
            # predicted_character = loaded_model.predict(dialogue_transformed)
            predicted_probabilities = loaded_model.predict_proba(dialogue_transformed)
            max_probability_index = np.argmax(predicted_probabilities)
            max_probability = predicted_probabilities[0][max_probability_index]
            predicted_character = loaded_model.classes_[max_probability_index]

            st.success(f"The predicted character is {predicted_character}")
            st.info(f"The confidence probability of the prediction is {max_probability:.2f}")
        
#-------------Visualization------------------------#
if selected_option == "Visualization":
    ## Data Visualization
    # Character lines
    character_lines = data['Character'].value_counts()
    # Define a list of colors
    colors = ["r", "purple", "b", "c", "m", "y"]
    
    query = st.selectbox("What visualization to show?", ["Visualize number of lines per character (Bar Chart)", "Number of lines per season", "Top characters per season", "Distribution of line lengths", "Average line lenth per character", "Sentiment analysis of dialogues", "About"])
    
    if query == "Visualize number of lines per character (Bar Chart)":
        character_lines = data['Character'].value_counts()
        top_characters = character_lines.head(10)
        plt.figure(figsize=(12, 6))
        plt.bar(top_characters.index, top_characters, color=colors)
        plt.xlabel('characters')
        plt.ylabel('Nunber of lines')
        plt.title('Number of lines per charater')
        st.pyplot(plt)
        
    elif query == "Number of lines per season":
        lines_per_season = data['Season'].value_counts().sort_index()
        plt.figure(figsize=(14, 8))
        plt.bar(lines_per_season.index, lines_per_season.values, color=colors)
        plt.xlabel('Season')
        plt.ylabel('Nunber of lines')
        plt.title('Number of lines per season')
        st.pyplot(plt)
    
    elif query == "Top characters per season":
        seasons = data['Season'].unique()
        character_lines = data['Character'].value_counts()
        top_characters = character_lines.head(10).index

        colors = ["r", "purple", "b", "c", "m", "y", "g", "orange", "brown"]

        plt.figure(figsize=(12,6))
        lefts = [0] * len(top_characters)

        for season, color in zip(seasons, colors):
            season_data = data[data['Season'] == season]
            character_lines_season = season_data["Character"].value_counts()
            top_characters_season = character_lines_season.loc[top_characters]

            plt.barh(top_characters_season.index, top_characters_season.values, left=lefts, color=color)
            lefts = [sum(x) for x in zip(lefts, top_characters_season.values)]

        plt.xlabel('Character')
        plt.ylabel('Nunber of lines')
        plt.title('Number of lines per character in season')
        plt.legend(seasons, title='Season')
        st.pyplot(plt)
    
    elif query == "Distribution of line lengths":
        data['Line_Length'] = data['Dialogue'].apply(lambda x: len(x.split()))
        plt.figure(figsize=(6, 2))
        plt.hist(data['Line_Length'], bins=3)
        plt.xlabel('Line length (words)')
        plt.ylabel('Frequency')
        plt.title('Distribution of line lengths')
        st.pyplot(plt)       
        
    elif query == "Average line lenth per character":
        character_lines = data['Character'].value_counts()
        top_characters = character_lines.head(10).index
        data['Line_Length'] = data['Dialogue'].apply(lambda x: len(x.split()))
        character_line_length = data.groupby('Character')['Line_Length'].mean().sort_values(ascending=False)
        top_character_line_length = character_line_length.loc[top_characters]
        plt.figure(figsize=(10, 5))
        plt.bar(top_character_line_length.index, top_character_line_length.values, color=colors)
        plt.xlabel('Character')
        plt.ylabel('Average line length (words)')
        plt.title('Average line length per character')
        st.pyplot(plt)   
    
    elif query == "Sentiment analysis of dialogues":
        def get_sentiment(text):
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        data['Sentiment'] = data['Dialogue'].apply(get_sentiment)
        plt.figure(figsize=(10, 5))
        plt.hist(data['Sentiment'], bins=10)
        plt.xlabel('Sentiment polarity')
        plt.ylabel('Frequency')
        plt.title(f'Sentiment analysis of dialogues')
        st.pyplot(plt) 
    
    elif query == "About":
        st.info("[Github repository](https://github.com/mregojos)")
    # Neutral 
    
#-------------Dataframe
if selected_option == "Script":
    st.dataframe(data)
