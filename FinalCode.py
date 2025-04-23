import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
#nltk.download('punkt', quiet=True)
#nltk.download('wordnet', quiet=True)

# Streamlit app title
st.title("Anomaly Detection in Logs")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',', on_bad_lines="skip", quoting=3, engine='python')
    df.drop(columns=['Time'], inplace=True, errors='ignore')
    df.rename(columns={"LineId": "Date", "Date": "time"}, inplace=True, errors="ignore")
    st.write("Data Preview:")
    st.dataframe(df)

    # Preprocess log messages
    def preprocess_logs(logs):
        lemmatizer = WordNetLemmatizer()
        custom_stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "must", "can", "cannot"}
        processed_logs = []
        for log in logs:
            tokens = word_tokenize(log.lower())
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in custom_stop_words]
            processed_logs.append(" ".join(tokens))
        return processed_logs
    
    # Check for message column
    if "message" in df.columns:
        logs = preprocess_logs(df["message"].astype(str).tolist())
    elif "Message" in df.columns:
        logs = preprocess_logs(df["Message"].astype(str).tolist())
    else:
        st.error("File must have a 'message' column.")
    
    # Convert logs to numerical representation
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(logs).toarray()
    
    # Function to determine optimal clusters
    def determine_optimal_clusters(X, max_clusters=5):
        best_sil_k = None
        best_db_k = None
        best_sil_score = -1  # Higher is better for Silhouette Score
        best_db_score = float("inf")  # Lower is better for Davies-Bouldin Score
        scores = []

        for k in range(2, max_clusters + 1):
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            
            scores.append((k, sil_score, db_score))
            
            # Update best Silhouette Score
            if sil_score > best_sil_score:
                best_sil_score = sil_score
                best_sil_k = k
            
            # Update best Davies-Bouldin Score
            if db_score < best_db_score:
                best_db_score = db_score
                best_db_k = k
        
        return best_sil_k, best_db_k, best_sil_score, best_db_score, scores
    
    best_sil_k, best_db_k, best_sil_score, best_db_score, cluster_scores = determine_optimal_clusters(X)

    # Compare which metric suggests the best number of clusters
    if best_sil_score > (1 / (1 + best_db_score)):
        optimal_k = best_db_k
        chosen_metric = "Davies-Bouldin Score"
    else:
        optimal_k = best_sil_k
        chosen_metric = "Silhouette Score"
    
    st.write(f"Silhouette Score: {best_sil_score}, Davies-Bouldin Score: {best_db_score}")
    st.write(f"Optimal number of clusters based on {chosen_metric}: {optimal_k}")

    # Apply clustering
    kmeans = MiniBatchKMeans(n_clusters=optimal_k, batch_size=1000, random_state=42, n_init='auto')
    df["Cluster"] = kmeans.fit_predict(X)

    # Reduce dimensionality using PCA
    pca = PCA(n_components=min(50, X.shape[0], X.shape[1]))
    X_reduced = pca.fit_transform(X)

    # Perform anomaly detection
    n_neighbors = st.slider("Neighbors for LOF", 3, 50, 5, step=1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    df["LOF_Anomaly"] = lof.fit_predict(X_reduced)

    # Plot clusters
    def plot_clusters(X, labels, anomaly_labels):
        pca = PCA(n_components=2)
        reduced_X = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=labels, palette="viridis", s=50)
        anomalies = (anomaly_labels == -1)
        plt.scatter(reduced_X[anomalies, 0], reduced_X[anomalies, 1], color='red', s=50, label="Anomalies", edgecolors='black')
        plt.xlabel("PC1:Maximum Variance In The Data")
        plt.ylabel("PC2:Next Highest Varience Perpendicular To PCA1")
        plt.title("Cluster Visualization with PCA")
        plt.legend(title="Legend")
        st.pyplot(plt)
    
    plot_clusters(X_reduced, labels=df["Cluster"].to_numpy(), anomaly_labels=df["LOF_Anomaly"].to_numpy())
    
    # Anomaly statistics
    num_records = max(len(df), 1)
    num_anomalies = np.sum(df["LOF_Anomaly"] == -1)
    anomaly_percentage = (num_anomalies / num_records) * 100
    
    st.write(f"Number of Logs In the Dataset: {num_records}")
    st.write(f"Number of Anomalies: {num_anomalies}")
    st.write(f"Percentage of anomalies: {anomaly_percentage:.2f}%")
    
    if "Cluster" in df.columns:
        silhouette_avg = silhouette_score(X_reduced, df["Cluster"])
        daviesbouldin_avg = davies_bouldin_score(X_reduced, df["Cluster"])
        st.write(f"Silhouette Score (higher is better): {silhouette_avg:.2f}")
        st.write(f"Davies-Bouldin Score (lower is better): {daviesbouldin_avg:.2f}")
    anomalies_df = df[df["LOF_Anomaly"] == -1]

    # Display anomalies in Streamlit
    st.subheader("Anomaly Messages")
    if not anomalies_df.empty:
        st.dataframe(anomalies_df)
        st.write(f"Total Anomalies Detected: {len(anomalies_df)}")
    else:
        st.write("No anomalies detected.")

    # Save the processed data
    df.to_csv("processed_logs.csv", index=False, encoding="utf-8-sig")
    # Read the file back into memory for download
    with open("processed_logs.csv", "rb") as f:
        st.download_button("Download Processed Logs", data=f, file_name="processed_logs.csv", mime="text/csv")
