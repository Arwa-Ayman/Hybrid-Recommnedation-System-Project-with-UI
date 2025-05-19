import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import warnings
import os
import requests
import zipfile
import io

warnings.filterwarnings('ignore')

# Download MovieLens 100K dataset if not present
@st.cache_data
def download_movielens():
    try:
        if not (os.path.exists('movies.csv') and os.path.exists('ratings.csv')):
            st.info("Downloading MovieLens 100K dataset...")
            url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
            response = requests.get(url)
            if response.status_code != 200:
                st.error("Failed to download dataset.")
                return None, None
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            zip_file.extract('ml-100k/u.item', 'data/')
            zip_file.extract('ml-100k/u.data', 'data/')
            movies = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1',
                                 names=['movieId', 'title', 'release_date', 'video_release', 'imdb_url'] + 
                                       ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                                        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
            # Convert binary genre columns to pipe-separated string
            genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            movies['genres'] = movies[genre_columns].apply(lambda x: '|'.join([col for col, val in x.items() if val == 1]), axis=1)
            movies = movies[['movieId', 'title', 'genres']]
            ratings = pd.read_csv('data/ml-100k/u.data', sep='\t',
                                  names=['userId', 'movieId', 'rating', 'timestamp'])
            return ratings, movies
        else:
            return load_data()
    except Exception as e:
        st.error(f"Error downloading dataset: {str(e)}")
        return None, None

# Data Ingestion and Preprocessing
@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv('ratings.csv', 
                            names=['userId', 'movieId', 'rating', 'timestamp'],
                            sep=',', skiprows=1)
        movies = pd.read_csv('movies.csv', 
                            names=['movieId', 'title', 'genres'],
                            sep=',', encoding='latin-1', skiprows=1)
        if ratings.empty or movies.empty:
            st.error("Error: One or both data files are empty.")
            return None, None
        invalid_titles = movies['title'].str.contains('Animation|Children|Romance|Drama|Unknown|Crime', case=False, na=False)
        if invalid_titles.any():
            st.warning(f"Found {invalid_titles.sum()} potentially invalid movie titles. Sample: {movies['title'][invalid_titles].head().tolist()}")
        return ratings, movies
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_data
def preprocess_data(ratings, movies):
    if ratings is None or movies is None:
        return None, None, None
    try:
        movies['title'] = movies['title'].fillna('Unknown')
        movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False).fillna('')
        comedy_count = movies['genres'].str.contains('Comedy', case=False, na=False).sum()
        if comedy_count == 0:
            st.warning(f"No movies found with 'Comedy' genre. Sample genres: {movies['genres'].head().tolist()}")
        if not pd.api.types.is_numeric_dtype(ratings['rating']):
            st.error(f"Ratings contain non-numeric values: {ratings['rating'].head().tolist()}")
            return None, None, None
        ratings['normalized_rating'] = (ratings['rating'] - 0.5) / (5 - 0.5)
        return ratings, movies, {'movie_count': len(movies), 'rating_count': len(ratings), 'user_count': ratings['userId'].nunique()}
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None, None

# Content-Based Filtering
@st.cache_resource
def setup_content_based(movies):
    try:
        tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['genres'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        return cosine_sim
    except Exception as e:
        st.error(f"Error setting up content-based filtering: {str(e)}")
        return None

def get_content_recommendations(movie_title, movies, cosine_sim, n=10):
    try:
        if movie_title not in movies['title'].values:
            st.error(f"Movie '{movie_title}' not found.")
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'content_score'])
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        indices = [i[0] for i in sim_scores]
        return movies.iloc[indices][['movieId', 'title', 'genres']].assign(content_score=[score for _, score in sim_scores])
    except Exception as e:
        st.error(f"Error in content-based recommendations: {str(e)}")
        return pd.DataFrame(columns=['movieId', 'title', 'genres', 'content_score'])

# Collaborative Filtering
@st.cache_resource
def setup_collaborative(ratings):
    try:
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        svd = SVD(random_state=42)
        svd.fit(trainset)
        predictions = svd.test(testset)
        return svd, predictions, testset
    except Exception as e:
        st.error(f"Error setting up collaborative filtering: {str(e)}")
        return None, None, None

def get_collab_recommendations(user_id, ratings, movies, svd, n=10):
    try:
        if user_id not in ratings['userId'].unique():
            st.error(f"User ID {user_id} not found.")
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'collab_score'])
        all_movie_ids = movies['movieId'].unique()
        rated = ratings[ratings['userId'] == user_id]['movieId'].values
        to_predict = np.setdiff1d(all_movie_ids, rated)
        if not to_predict.size:
            st.warning(f"No unrated movies for user {user_id}.")
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'collab_score'])
        testset = [[user_id, mid, 4.0] for mid in to_predict]
        preds = svd.test(testset)
        top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:n]
        top_ids = [pred.iid for pred in top_preds]
        return movies[movies['movieId'].isin(top_ids)][['movieId', 'title', 'genres']].assign(collab_score=[pred.est for pred in top_preds])
    except Exception as e:
        st.error(f"Error in collaborative recommendations: {str(e)}")
        return pd.DataFrame(columns=['movieId', 'title', 'genres', 'collab_score'])

# Hybrid Recommendations
def hybrid_recommendations(user_id, movie_title, ratings, movies, svd, cosine_sim, n=10, content_w=0.5, collab_w=0.5):
    try:
        content_recs = get_content_recommendations(movie_title, movies, cosine_sim, n)
        collab_recs = get_collab_recommendations(user_id, ratings, movies, svd, n)
        if content_recs.empty and collab_recs.empty:
            st.error("Both content-based and collaborative recommendations failed.")
            return pd.DataFrame()
        merged_recs = pd.concat([content_recs, collab_recs], ignore_index=True).drop_duplicates(subset='movieId')
        if merged_recs.empty:
            st.error("No recommendations after merging.")
            return pd.DataFrame()
        scaler = MinMaxScaler()
        merged_recs[['content_score', 'collab_score']] = scaler.fit_transform(merged_recs[['content_score', 'collab_score']].fillna(0))
        merged_recs['hybrid_score'] = content_w * merged_recs['content_score'] + collab_w * merged_recs['collab_score']
        return merged_recs[['movieId', 'title', 'genres', 'hybrid_score']].sort_values(by='hybrid_score', ascending=False).head(n)
    except Exception as e:
        st.error(f"Error in hybrid recommendations: {str(e)}")
        return pd.DataFrame()

# Evaluation Metrics
def evaluate_model(ratings, predictions, testset, k=10, threshold=3.5):
    try:
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        user_est_true = {}
        for pred in predictions:
            user_id = pred.uid
            if user_id not in user_est_true:
                user_est_true[user_id] = []
            user_est_true[user_id].append((pred.est, pred.r_ui))
        precisions, recalls = [], []
        for user_id, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
            precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
            recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return {'rmse': rmse, 'mae': mae, 'precision': precision, 'recall': recall, 'f1': f1}
    except Exception as e:
        st.error(f"Error in model evaluation: {str(e)}")
        return {'rmse': np.nan, 'mae': np.nan, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

# Streamlit UI
def main():
    st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
    st.title("🎬 Hybrid Movie Recommendation System")

    # Load or download data
    ratings, movies = download_movielens()
    if ratings is None or movies is None:
        return
    ratings, movies, stats = preprocess_data(ratings, movies)
    if ratings is None or movies is None:
        return
    cosine_sim = setup_content_based(movies)
    svd, predictions, testset = setup_collaborative(ratings)
    if cosine_sim is None or svd is None:
        return
    metrics = evaluate_model(ratings, predictions, testset)

    # Sidebar navigation
    app_mode = st.sidebar.radio("Choose mode", ["Home", "Content-Based", "Collaborative", "Hybrid", "Evaluation"])

    if app_mode == "Home":
        st.markdown("## MovieLens 100K Dataset Overview")
        st.write(f"Total Movies: {stats['movie_count']}")
        st.write(f"Total Ratings: {stats['rating_count']}")
        st.write(f"Total Users: {stats['user_count']}")
        st.write("Sample Movies:")
        st.dataframe(movies[['movieId', 'title', 'genres']].head(5))
    elif app_mode == "Content-Based":
        st.subheader("Content-Based Recommendations")
        movie = st.selectbox("Choose a movie", sorted(movies['title'].unique()))
        n_recs = st.slider("Number of Recommendations", 1, 20, 10)
        if st.button("Get Recommendations"):
            recs = get_content_recommendations(movie, movies, cosine_sim, n_recs)
            if not recs.empty:
                st.dataframe(recs)
            else:
                st.error("No recommendations available.")
    elif app_mode == "Collaborative":
        st.subheader("Collaborative Filtering Recommendations")
        user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings['userId'].max(), value=1)
        n_recs = st.slider("Number of Recommendations", 1, 20, 10)
        if st.button("Get Recommendations"):
            recs = get_collab_recommendations(user_id, ratings, movies, svd, n_recs)
            if not recs.empty:
                st.dataframe(recs)
            else:
                st.error("No recommendations available.")
    elif app_mode == "Hybrid":
        st.subheader("Hybrid Recommendations")
        user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings['userId'].max(), value=1)
        movie = st.selectbox("Choose a movie", sorted(movies['title'].unique()))
        content_w = st.slider("Content-Based Weight", 0.0, 1.0, 0.5)
        collab_w = st.slider("Collaborative Weight", 0.0, 1.0, 0.5)
        n_recs = st.slider("Number of Recommendations", 1, 20, 10)
        if st.button("Get Recommendations"):
            recs = hybrid_recommendations(user_id, movie, ratings, movies, svd, cosine_sim, n_recs, content_w, collab_w)
            if not recs.empty:
                st.dataframe(recs)
            else:
                st.error("No recommendations available.")
    elif app_mode == "Evaluation":
        st.subheader("Model Performance")
        st.write(f"RMSE: {metrics['rmse']:.4f}")
        st.write(f"MAE: {metrics['mae']:.4f}")
        st.write(f"Precision@10: {metrics['precision']:.4f}")
        st.write(f"Recall@10: {metrics['recall']:.4f}")
        st.write(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()