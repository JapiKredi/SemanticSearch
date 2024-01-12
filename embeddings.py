import numpy as pd
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
import matplotlib.pyplot as plt

df = pd.read_csv('Food App Reviews.csv')

reviews = df['Review'].tolist()

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Generate sentence embeddings
embeddings = model.encode(reviews)

