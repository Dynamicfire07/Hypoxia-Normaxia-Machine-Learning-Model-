import os
os.environ["SCIPY_ARRAY_API"] = "1"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

# Load all files
shZHX2 = pd.read_csv('datasets/GSE260794_shZHX2-raw-count.txt', sep='\t')
sgZHX2 = pd.read_csv('datasets/GSE260794_sgZHX2-raw-count.txt', sep='\t')
rescue = pd.read_csv('datasets/GSE260794_rescue-ZHX2-raw-count.txt', sep='\t')
hd = pd.read_csv('datasets/GSE260794_1.6HD-raw-count.txt', sep='\t')

# Rename count columns for clarity
shZHX2 = shZHX2.rename(columns={shZHX2.columns[1]: 'shZHX2'})
sgZHX2 = sgZHX2.rename(columns={sgZHX2.columns[1]: 'sgZHX2'})
rescue = rescue.rename(columns={rescue.columns[1]: 'rescue'})
hd = hd.rename(columns={hd.columns[1]: '1.6HD'})

# Merge on GeneID
merged = shZHX2.merge(sgZHX2, on='Geneid') \
                .merge(rescue, on='Geneid') \
                .merge(hd, on='Geneid')

# Show preview
print(merged.head())
print(shZHX2.head())
print(shZHX2.columns)
merged = merged.fillna(0)
merged = merged.dropna()
merged = merged.drop_duplicates(subset='Geneid')
merged = merged.set_index('Geneid')
merged = merged.apply(pd.to_numeric, errors='coerce')
merged = merged.fillna(0)
merged = merged[merged.sum(axis=1) >= 10]
print(merged.shape)
print(merged.head())
# CPM-like normalization
normalized = merged.div(merged.sum(axis=0), axis=1) * 1e6
log_transformed = np.log1p(normalized)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_array = scaler.fit_transform(log_transformed.T).T  # transpose so genes are features
scaled_df = pd.DataFrame(scaled_array, index=log_transformed.index, columns=log_transformed.columns)
# Calculate variance
gene_variances = scaled_df.var(axis=1)

# Select top 2000 most variable genes (you can choose N)
top_genes = gene_variances.sort_values(ascending=False).head(2000).index

# Filter
selected_df = scaled_df.loc[top_genes]
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Transpose: samples as rows, genes as columns
pca = PCA(n_components=2)
pca_result = pca.fit_transform(selected_df.T)

# Plot
plt.scatter(pca_result[:,0], pca_result[:,1])
plt.title('PCA of Samples')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(selected_df.T)

# Add to PCA plot
plt.scatter(pca_result[:,0], pca_result[:,1], c=labels)
plt.title('PCA with KMeans Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Select top 50 most variable genes for clarity
top_50_genes = selected_df.var(axis=1).sort_values(ascending=False).head(50).index
heatmap_data = selected_df.loc[top_50_genes]

plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_data, 
    cmap='viridis', 
    xticklabels=True, 
    yticklabels=True, 
    cbar_kws={'label': 'Standardized Expression'}
)
plt.title('Gene Expression Heatmap (Top 50 High-Variance Genes)')
plt.xlabel('Samples')
plt.ylabel('Genes')
plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1️⃣ Transpose so samples = rows
X = selected_df.T
print("✅ X shape:", X.shape)
print("✅ Sample names example:", X.index.tolist()[:10])

# 2️⃣ Build binary labels for hypoxia vs normoxia
labels = []
for name in X.index:
    name_lower = name.lower()
    if any(key in name_lower for key in ["h.", "h_", "h-", "hd", "1.6hd"]):
        labels.append("hypoxia")
    elif any(key in name_lower for key in ["n.", "ctrl", "control", "nc"]):
        labels.append("normoxia")
    else:
        labels.append("unknown")

# Filter unknown samples
mask = [lab != "unknown" for lab in labels]
X = X[mask]
y = [lab for lab in labels if lab != "unknown"]

print("✅ Labeled samples:", {l: y.count(l) for l in set(y)})

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

# Pipeline with PCA for dimensionality reduction and logistic regression
n_components = min(10, X_train.shape[0] - 1)
model = Pipeline([
    ("pca", PCA(n_components=n_components)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Cross-validation on the training set
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"✅ Cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit on the full training set and evaluate on the hold-out test set
model.fit(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("✅ Test accuracy:", test_accuracy)
