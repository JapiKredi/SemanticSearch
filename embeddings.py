import numpy as pd
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import plotly.express as px

df = pd.read_csv('Food App Reviews.csv')

reviews = df['Review'].tolist()

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Generate sentence embeddings
embeddings = model.encode(reviews)

# Generate tensor embeddings
embeddings = model.encode(reviews, convert_to_tensor=True)

# Reduce dimensionality with UMAP
reducer = UMAP(n_components = 2, metric='cosine')
embeddings_2d = reducer.fit_transform(embeddings)

# Plot the embeddings
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker = 'o')
plt.title('Food App Review Embeddings')
plt.show()

print('line 32')
# Apply K-means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(embeddings_2d)

print('line 38')

# Add cluster labels as a new column to the DataFrame
df['cluster_label'] = cluster_labels

print('line 43')

# Create a DataFrame for the data
cluster_df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'label': cluster_labels,
    'sentence': df['Review']
})

print('line 53')

# Create an interactive scatter plot using plotly
fig = px.scatter(
    cluster_df,
    x='x', y='y',
    color='label',
    hover_name='sentence',
    title='Food App Reviews 2D Embeddings',
    labels={'label': 'Cluster'},
    width=800,  # Adjust the width as desired
    height=600,  # Adjust the height as desired
)

print('line 67')

fig.update_traces(
    marker=dict(size=8)  # Adjust the size value as needed
)

print('line 73')

# Set the background color to black
fig.update_layout(
    plot_bgcolor='white',
)

fig.show()

print('line 82')

# Reduce dimensionality to 3 dimensions with UMAP
reducer = UMAP(n_components=3, metric='cosine')
embeddings_3d = reducer.fit_transform(embeddings)

print('line 88')

# Apply K-means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(embeddings)

print('line 95')

# Add cluster labels as a new column to the DataFrame
df['cluster_label'] = cluster_labels

print('line 100')

# Create a DataFrame for the data
cluster_df = pd.DataFrame({
    'x': embeddings_3d[:, 0],
    'y': embeddings_3d[:, 1],
    'z': embeddings_3d[:, 2],
    'label': cluster_labels,
    'sentence': df['Review']
})

print('line 111')

# Create a 3D scatter plot using plotly
fig = px.scatter_3d(
    cluster_df,
    x='x', y='y', z='z',
    color='label',
    hover_name='sentence',
    title='Food App Reviews 3D Embeddings',
    labels={'label': 'Cluster'},
)

print('line 123')

fig.update_traces(
    marker=dict(size=5)  # Adjust the size value as needed
)

fig.show()
print('line 130')