# train_models.py
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Dataset tiruan
data = pd.DataFrame({
    'usia': [18, 20, 22, 19, 21, 24, 23, 17],
    'jenis_kelamin': ['Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Pria'],
    'pendidikan': ['SMA', 'Sarjana', 'Pascasarjana', 'SMA', 'Sarjana', 'SMA', 'Sarjana', 'SMA'],
    'pekerjaan_ortu': ['Guru', 'Dokter', 'Petani', 'Guru', 'PNS', 'Petani', 'PNS', 'Dokter'],
    'penghasilan_ortu': [3e6, 8e6, 2e6, 3.5e6, 7e6, 1.5e6, 6e6, 5e6],
    'biaya_pendidikan': [5e6, 15e6, 20e6, 5.5e6, 14e6, 4e6, 13e6, 12e6]
})

# Label untuk klasifikasi
data['label_biaya'] = pd.cut(data['biaya_pendidikan'], bins=[0, 7e6, 13e6, float('inf')],
                             labels=["Rendah", "Sedang", "Tinggi"])

X = data.drop(['biaya_pendidikan', 'label_biaya'], axis=1)
y_reg = data['biaya_pendidikan']
y_clf = data['label_biaya']

categorical = ['jenis_kelamin', 'pendidikan', 'pekerjaan_ortu']
numeric = ['usia', 'penghasilan_ortu']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical)
], remainder='passthrough')

# Regresi
reg_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
reg_model.fit(X, y_reg)
joblib.dump(reg_model, 'model_regresi.pkl')

# Klasifikasi
clf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
clf_model.fit(X, y_clf)
joblib.dump(clf_model, 'model_klasifikasi.pkl')

# Clustering
X_clust = preprocessor.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X_clust)
joblib.dump((preprocessor, kmeans), 'model_klaster.pkl')
