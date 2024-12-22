
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data
df = pd.read_csv('BC.csv')

# Preprocessing
duplicate = df.duplicated().sum()
null_values = df.isnull().sum()

# Split data
y = df['diagnosis']
X = df.drop(columns=['diagnosis'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Logistic Regression with GridSearchCV
param_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]}
log_reg_tuned = LogisticRegression()
grid_search = GridSearchCV(estimator=log_reg_tuned, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Use the best model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

# Intercept and coefficients
intercept_tuned = best_model.intercept_[0]
coefficients_tuned = best_model.coef_[0]

# Correlation check
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop highly correlated features
X_train_dropped = X_train.drop(columns=to_drop)
X_test_dropped = X_test.drop(columns=to_drop)

# Add constant for statsmodels
X_const_tuned = sm.add_constant(X_train_dropped)
logit_model = sm.Logit(y_train, X_const_tuned)
result = logit_model.fit()

conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)
class_report = classification_report(y_test, y_pred_tuned, output_dict=True)
y_pred_prob = result.predict(X_const_tuned)

# K-Means Clustering
aggregated_df = df.groupby('id').agg(
    radius_mean=('radius_mean', 'mean'),
    texture_mean=('texture_mean', 'mean'),
    perimeter_mean=('perimeter_mean', 'mean'),
    area_mean=('area_mean', 'mean'),
    smoothness_mean=('smoothness_mean', 'mean'),
    compactness_mean=('compactness_mean', 'mean'),
    concavity_mean=('concavity_mean', 'mean'),
    concave_points_mean=('concave_points_mean', 'mean'),
    symmetry_mean=('symmetry_mean', 'mean'),
    fractal_dimension_mean=('fractal_dimension_mean', 'mean'),
    radius_worst=('radius_worst', 'max'),
    texture_worst=('texture_worst', 'max'),
    perimeter_worst=('perimeter_worst', 'max'),
    area_worst=('area_worst', 'max'),
    smoothness_worst=('smoothness_worst', 'max'),
    compactness_worst=('compactness_worst', 'max'),
    concavity_worst=('concavity_worst', 'max'),
    concave_points_worst=('concave_points_worst', 'max'),
    symmetry_worst=('symmetry_worst', 'max'),
    fractal_dimension_worst=('fractal_dimension_worst', 'max')
).reset_index()

max_k = min(14, len(aggregated_df))
inertia = []
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(aggregated_df)
    inertia.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(aggregated_df)
aggregated_df['cluster'] = kmeans.labels_

# PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(aggregated_df.drop(columns=['id', 'cluster']))
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = aggregated_df['cluster'].replace({0: 'Tidak Ganas', 1: 'Ganas'})
centroids = pca_df.groupby('Cluster').mean()

# Streamlit Sidebar
st.sidebar.title("Segmentasi Analisis")
option = st.sidebar.radio("Pilih Analisis:", [
    "Dataset Info",
    "Preprocessing Data", 
    "Distribusi Data", 
    "Logistic Regression", 
    "K-Means Clustering", 
    "Prediksi Diagnosa"])

# Streamlit Dashboard
if option == "Dataset Info":
    st.title("Informasi Dataset")
    st.write("**Deskripsi Dataset:**")
    st.write("Dataset ini berisi data untuk analisis diagnosis kanker payudara. Setiap baris dalam dataset mewakili pengukuran tertentu dari pasien.")
    st.write("**Jumlah Data dan Fitur:**")
    st.write(f"- Jumlah baris (data): {df.shape[0]}")
    st.write(f"- Jumlah kolom (fitur): {df.shape[1]}")
    st.write("Isi keseluruhan data breast cancer:")
    st.dataframe(df)
    st.write("**Daftar Fitur/Kolom:**")
    st.dataframe(pd.DataFrame(df.columns, columns=["Fitur/Kolom"]))

elif option == "Preprocessing Data":
    st.title("Data Preprocessing")
    st.write(f"Jumlah data yang duplikat: {duplicate}")
    st.write("Rincian Kolom dalam Dataset:")
    data_info = pd.DataFrame({
        'Nama Kolom': df.columns,
        'Tipe Data': df.dtypes,
        'Jumlah Null': df.isnull().sum(),
        'Contoh Data': [df[col].iloc[0] for col in df.columns]
    })
    st.dataframe(data_info)
    st.write("Statistik Deskriptif Kolom Numerik:")
    st.dataframe(df.describe().T)

elif option == "Distribusi Data":
    st.title("Distribusi Data")
    st.write("Distribusi Diagnosis:")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='diagnosis', palette='Set2')
    plt.title('Distribusi Diagnosis', fontsize=16)
    plt.xlabel('Diagnosis (0: Tidak Ganas, 1: Ganas)', fontsize=12)
    plt.ylabel('Jumlah', fontsize=12)
    st.pyplot(plt)

    st.write("Distribusi Kolom Numerik:")
    fig, ax = plt.subplots(figsize=(20, 15))
    df.iloc[:, 2:].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
    plt.suptitle('Distribusi Kolom Numerik', fontsize=16)
    st.pyplot(fig)

    st.write("Matriks Korelasi Antar Fitur:")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.iloc[:, 2:].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Korelasi'})
    plt.title('Matriks Korelasi Antar Fitur', fontsize=16)
    st.pyplot(plt)

    st.write("Distribusi Fitur Berdasarkan Diagnosis:")
    plt.figure(figsize=(12, 6))
    for col in ['radius_mean', 'texture_mean', 'perimeter_mean']:
        sns.kdeplot(data=df, x=col, hue='diagnosis', fill=True)
    plt.title('Distribusi Berdasarkan Diagnosis', fontsize=16)
    plt.xlabel('Nilai Fitur', fontsize=12)
    plt.ylabel('Kepadatan', fontsize=12)
    st.pyplot(plt)

elif option == "Logistic Regression":
    st.title("Logistic Regression Model")
    st.write("""**Logistic Regression:** Menggunakan GridSearchCV untuk mencari parameter terbaik (C).
    Evaluasi dilakukan menggunakan akurasi, confusion matrix, dan classification report.
    """)
    
    # Show the best parameters and model accuracy
    st.write(f"**- Best Parameters:** {best_params}")
    st.write(f"**- Best Score (Cross-Validation):** {best_score}")
    st.write(f"**- Tuned Model Accuracy on Test Data:** {accuracy_tuned:.2f}")
    st.write(f"**- Tuned Model Intercept:** {intercept_tuned:.2f}")
    st.write(f"**- Tuned Model Coefficients:** {coefficients_tuned}")

    # Plot predicted probabilities
    y_pred_prob = result.predict(X_const_tuned)
    threshold = 0.5

    st.write("**- Histogram Breast Cancer Level Probabilities** ")
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_prob[y_train == 1], bins=30, alpha=0.7, label='Ganas', color='red')
    plt.hist(y_pred_prob[y_train == 0], bins=30, alpha=0.7, label='Tidak Ganas', color='blue')
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Logit Model Predicted Probabilities After Tuning')
    plt.legend()
    st.pyplot(plt)

    # Confusion Matrix and Classification Report
    st.write("Confusion Matrix (After Tuning):")
    st.dataframe(conf_matrix_tuned)

    st.write("Classification Report (After Tuning):")
    st.write("**Precision, Recall, F1-Score, and Support for Each Class:**")
    
    # Format and display the classification report neatly in a table
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df)

    st.write("P-Values from Logistic Regression (After Removing Correlated Features):")
    st.write(result.pvalues)
elif option == "K-Means Clustering":
    st.title("K-Means Clustering")
    st.write("""
    **K-Means Clustering:**
    - Algoritma unsupervised learning untuk mengelompokkan data menjadi 2 cluster (Ganas/Tidak Ganas).
    - Menggunakan Elbow Method untuk menentukan jumlah cluster optimal.
    - PCA digunakan untuk visualisasi hasil clustering dalam 2 dimensi.
    """)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette={'Tidak Ganas': 'green', 'Ganas': 'red'})
    plt.scatter(centroids['PCA1'], centroids['PCA2'], s=100, c='black', label='Centroids', marker='X')
    plt.title('K-Means Clustering Breast Cancer Results')
    plt.legend()
    st.pyplot(plt)

elif option == "Prediksi Diagnosa":
    st.title("Prediksi Diagnosa")
    st.write("Masukkan Data Pasien:")

    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(f"{column}", value=0.0)

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        prediction = best_model.predict(input_df)[0]
        result = "Ganas" if prediction == 1 else "Tidak Ganas"
        st.write(f"Hasil Prediksi: {result}")

# Save models
with open('modelLog.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('kmeans_model.pkl', 'wb') as f_kmeans:
    pickle.dump(kmeans, f_kmeans)
