import streamlit as st
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer



# Preprocessing and Model Building

# Loading data
df = pd.read_csv("./dataset/anime_with_synopsis.csv")
df.dropna(inplace=True)
df["Score"] = df["Score"].map(lambda x:np.nan if x=="Unknown" else x)
df["Score"].fillna(df["Score"].median(),inplace = True)
df["Score"] = df["Score"].astype(float)
df['Genres'] = df['Genres'].apply(lambda x:x.split())
df['sypnopsis'] = df['sypnopsis'].apply(lambda x:x.split())
df['Genres'] = df['Genres'].apply(lambda x:[i.replace(" ","") for i in x])
df['sypnopsis'] = df['sypnopsis'].apply(lambda x:[i.replace(" ","") for i in x])
df['features'] = df['Genres'] + df['sypnopsis']
new_df = df[['Name', 'features']]

# Stemming and vectorizing
ps = PorterStemmer()
def stem(text):
    y = []
    
    # check if the text is a string
    if isinstance(text, str):
        text = text.split()

    # iterate over the words
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

new_df['features'] = new_df['features'].apply(stem)
new_df['features'] = new_df['features'].apply(lambda x:x.lower())
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['features']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Recommendation Function
def recommend(anime):
    movie_index = new_df[new_df['Name'] == anime].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:15]
    recommended_anime = []
    for i in movies_list:
        anime_name = new_df.iloc[i[0]].Name
        anime_score = df[df['Name'] == anime_name]['Score'].values[0] 
        anime_genres = df[df['Name'] == anime_name]['Genres'].values[0]
        anime_synopsis = df[df['Name'] == anime_name]['sypnopsis'].values[0]
        recommended_anime.append((anime_name, anime_score, anime_genres, anime_synopsis))
    recommended_anime = sorted(recommended_anime, key=lambda x:x[1], reverse=True)
    return recommended_anime[:10]

# Streamlit App
from PIL import Image

from PIL import Image

def main():
    
    # Load the image
    image = Image.open('image.jpeg')

    # Create columns
    col1, col2, col3 = st.columns([1,2,1])

    # Display the image in the middle column
    with col2:
        st.image(image, use_column_width=True)
    
    st.title("Anime Recommendation System")
    favourite_anime = st.text_input("Enter your favourite anime")
    if st.button("Recommend"):
        result = recommend(favourite_anime)
        for i in range(10):
            st.write("Name: ", result[i][0])
            st.write("Score: ", result[i][1])
            st.write("Genres: ", ' '.join(result[i][2]))
            st.write("Synopsis: ", ' '.join(result[i][3]))
            st.write("---")

if __name__ == '__main__':
    main()


