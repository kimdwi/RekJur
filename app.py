# app.py

from flask import Flask, render_template, request
from model import get_jurusan_rekomendasi

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("input-nilai.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        minat = request.form['minat']
        bakat = request.form['bakat']
        kemampuan = request.form['kemampuan']

        # Mendapatkan rekomendasi jurusan berdasarkan inputan
        jurusan_rekomendasi = get_jurusan_rekomendasi(minat, bakat, kemampuan)

        # Render halaman hasil
        return render_template('predict.html', jurusan_rekomendasi=jurusan_rekomendasi)
    else:
        # Handle case when the method is not POST
        return render_template('input-nilai.html')

if __name__ == "__main__":
    app.run(debug=True)
