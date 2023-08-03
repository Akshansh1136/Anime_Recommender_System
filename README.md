# Anime Recommender System


This is a content-based Anime Recommendation System, developed using Python and Streamlit, with Natural Language Processing (NLP) techniques for pre-processing. This project provides a user-friendly interface where users can input their favorite anime and get the top-10 similar anime recommendations in return.

# Project Description
The recommendation system uses content-based filtering, a fundamental technique in recommender systems. It processes textual data such as genres and synopses associated with each anime using tokenization and stemming, essential NLP techniques. The textual features are then converted into numerical vectors using CountVectorizer from sklearn. These vectors are assessed using cosine similarity to measure the similarity between various animes, forming the basis of the recommendation mechanism.

# Getting Started

## Prerequisites
The project uses Python 3.8.5 and the following libraries should be installed to ensure the project runs successfully:

- pandas
- numpy
- nltk
- sklearn
- streamlit
- PIL

`pip install pandas numpy nltk sklearn streamlit pillow`
## Dataset

The project uses a dataset in CSV format, which can be found in the ./anime_with_synopsis.csv file.

## Running the Application

You can run the application using Streamlit. Navigate to the project's root directory in your terminal and run the following command:

`streamlit run app.py`

## Usage

- Upon launching the application, you'll see an interface asking for your favorite anime.
- Enter the name of the anime in the provided text box.
- Click on the "Recommend" button.
- The system will display the top-10 similar animes, along with their respective scores, genres, and synopses.

## Project Structure
This project has the following structure:

- `./anime_with_synopsis.csv`: This csv file contains the dataset used in this project.
- `anime_recommendation_system.ipynb`: This is the Jupyter notebook where the model is developed.
- `app.py`: This is the main Python script where the web application is created using Streamlit.


