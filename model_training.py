import os
os.environ["SCIPY_ARRAY_API"] = "1"
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

files = {
    'shZHX2': 'datasets/GSE260794_shZHX2-raw-count.txt',
    'sgZHX2': 'datasets/GSE260794_sgZHX2-raw-count.txt',
    'rescue': 'datasets/GSE260794_rescue-ZHX2-raw-count.txt',
    'hd': 'datasets/GSE260794_1.6HD-raw-count.txt'
}

def load_rename(path, label):
    df = pd.read_csv(path, sep='\t')
    df = df.rename(columns={df.columns[1]: label})
    return df

shZHX2 = load_rename(files['shZHX2'], 'shZHX2')
sgZHX2 = load_rename(files['sgZHX2'], 'sgZHX2')
rescue = load_rename(files['rescue'], 'rescue')
hd = load_rename(files['hd'], '1.6HD')

merged = shZHX2.merge(sgZHX2, on='Geneid').merge(rescue, on='Geneid').merge(hd, on='Geneid')
merged = merged.drop_duplicates('Geneid').set_index('Geneid')
merged = merged.apply(pd.to_numeric, errors='coerce').fillna(0)
merged = merged[merged.sum(axis=1) >= 10]

# CPM normalization and log transform
normalized = merged.div(merged.sum(axis=0), axis=1) * 1e6
log_transformed = np.log1p(normalized)

# Standardize genes
scaler = StandardScaler()
scaled_array = scaler.fit_transform(log_transformed.T).T
scaled_df = pd.DataFrame(scaled_array, index=log_transformed.index, columns=log_transformed.columns)

# Feature selection: top 100 most variable genes
variances = scaled_df.var(axis=1)
top_genes = variances.sort_values(ascending=False).head(100).index
selected_df = scaled_df.loc[top_genes]

# Build sample matrix
X = selected_df.T

# Label samples
labels = []
for name in X.index:
    name_lower = name.lower()
    if any(k in name_lower for k in ['h.', 'h_', 'h-', 'hd', '1.6hd']):
        labels.append('hypoxia')
    else:
        labels.append('normoxia')

y = np.array([1 if l == 'hypoxia' else 0 for l in labels])

smote = SMOTE()
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', penalty='l2')

pipeline = ImbPipeline([
    ('smote', smote),
    ('pca', PCA(n_components=min(10, X.shape[0] - 1))),
    ('clf', log_reg)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy').mean()
print(f"{accuracy*100:.2f}%")
