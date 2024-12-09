import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure Streamlit
st.set_page_config(
    page_title="Weather Forecasting with Clustering",
    page_icon="☁️",
    layout="wide"
)

# Title and description
st.title("🌦️ Weather Forecasting with Pattern Clustering")
st.write("Analyze and cluster weather patterns to observe climate trends.")

# File upload
uploaded_file = st.file_uploader("Upload a weather dataset (CSV)", type="csv")

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Select numeric columns for clustering
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    selected_columns = st.multiselect(
        "Select columns for clustering", numeric_columns, default=numeric_columns[:3]
    )

    if selected_columns:
        # Preprocessing
        X = data[selected_columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Number of clusters
        st.write("### Number of Clusters")
        n_clusters = st.slider("Select the number of clusters", 2, 10, 3)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        data["Cluster"] = clusters

        # Cluster visualization
        st.write("### Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=X[selected_columns[0]],
            y=X[selected_columns[1]],
            hue=clusters,
            palette="viridis",
            s=100,
            ax=ax
        )
        plt.title("Clusters of Weather Patterns")
        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1])
        st.pyplot(fig)

        # Pairplot
        st.write("### Pairplot of Clusters")
        pairplot_fig = sns.pairplot(data, vars=selected_columns, hue="Cluster", palette="bright")
        st.pyplot(pairplot_fig)

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        corr = data[selected_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Cluster Statistics
        st.write("### Cluster Statistics")
        cluster_summary = data.groupby("Cluster")[selected_columns].mean()
        st.write(cluster_summary)
else:
    st.write("Please upload a dataset to begin.")

# Footer
st.write("Developed by Climate Analytics Team")
