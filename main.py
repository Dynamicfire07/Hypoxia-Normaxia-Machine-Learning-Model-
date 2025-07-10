import os
os.environ["SCIPY_ARRAY_API"] = "1"
import pandas as pd
import numpy as np


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

merged = merged.fillna(0)
merged = merged.dropna()
merged = merged.drop_duplicates(subset='Geneid')
merged = merged.set_index('Geneid')
merged = merged.apply(pd.to_numeric, errors='coerce')
merged = merged.fillna(0)
# Filter lowly expressed genes
merged = merged[merged.sum(axis=1) >= 10]
# CPM-like normalization
normalized = merged.div(merged.sum(axis=0), axis=1) * 1e6
log_transformed = np.log1p(normalized)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_array = scaler.fit_transform(log_transformed.T).T  # transpose so genes are features
scaled_df = pd.DataFrame(scaled_array, index=log_transformed.index, columns=log_transformed.columns)
# Calculate variance
gene_variances = scaled_df.var(axis=1)

# Select top 100 most variable genes
top_genes = gene_variances.sort_values(ascending=False).head(100).index

# Filter
selected_df = scaled_df.loc[top_genes]
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1️⃣ Transpose so samples = rows
X = selected_df.T

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

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Stratified K-fold with oversampling
min_class = np.bincount(y_enc).min()
n_splits = min(5, min_class)
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy")
print(f"{scores.mean()*100:.2f}")
