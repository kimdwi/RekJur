from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

siswa_df = pd.read_csv('dataset/siswa.csv', sep=";")
jurusan_df = pd.read_csv('dataset/jurusan.csv', sep=";")
label_df = pd.read_csv('dataset/label.csv', sep=";")

# Create label_to_jurusan_rf mapping
label_to_jurusan_rf = dict(zip(label_df['Label'], label_df['Jurusan']))

# Merge dataset siswa, jurusan, dan label
merged_df = pd.merge(siswa_df, label_df, how="inner", left_on="Jurusan", right_on="Jurusan")
final_df = pd.merge(merged_df, jurusan_df, how="inner", left_on="Jurusan", right_on="Jurusan")

# Perubahan nama kolom setelah di merge
final_df.rename(columns={"Minat_x": "Minat", "Bakat_x": "Bakat", "Kemampuan_x": "Kemampuan"}, inplace=True)

# Persiapkan data untuk model
X = final_df[["Minat", "Bakat", "Kemampuan"]].values
y = final_df["Label"].values

# Normalisasi data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Model Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Function untuk mendapatkan rekomendasi jurusan berdasarkan inputan
def get_jurusan_rekomendasi(minat, bakat, kemampuan):
    siswa_normalized = scaler.transform(np.array([minat, bakat, kemampuan]).reshape(1, -1))
    predicted_probabilities_rf = model_rf.predict_proba(siswa_normalized)
    
    # Urutkan jurusan berdasarkan probabilitas tertinggi
    sorted_jurusan_indices_rf = np.argsort(predicted_probabilities_rf[0])[::-1]

    # Tampilkan beberapa rekomendasi teratas
    num_rekomendasi_rf = 3  # Sesuai kebutuhan
    top_rekomendasi_rf = sorted_jurusan_indices_rf[:num_rekomendasi_rf]

    # Output hasil dalam bentuk nama jurusan beserta probabilitas (%)
    jurusan_rekomendasi = []
    for index in top_rekomendasi_rf:
        label_rf = index
        jurusan_rf = label_to_jurusan_rf[label_rf]
        probabilitas_rf = predicted_probabilities_rf[0, label_rf]
        jurusan_rekomendasi.append((jurusan_rf, probabilitas_rf))
    
    return jurusan_rekomendasi

# Rute untuk menampilkan halaman input nilai
@app.route('/input-nilai', methods=['GET', 'POST'])
def input_nilai():
    if request.method == 'POST':
        minat = request.form['minat']
        bakat = request.form['bakat']
        kemampuan = request.form['kemampuan']

        # Mendapatkan rekomendasi jurusan berdasarkan inputan
        jurusan_rekomendasi = get_jurusan_rekomendasi(minat, bakat, kemampuan)

        # Render halaman hasil
        return render_template('predict.html', jurusan_rekomendasi=jurusan_rekomendasi)
    else:
        return render_template('input-nilai.html')
