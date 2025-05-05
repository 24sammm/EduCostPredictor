from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
reg_model = joblib.load('model_regresi.pkl')
clf_model = joblib.load('model_klasifikasi.pkl')
preprocessor_clust, clust_model = joblib.load('model_klaster.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    model_used = None
    if request.method == 'POST':
        usia = int(request.form['usia'])
        jenis_kelamin = request.form['jenis_kelamin']
        pendidikan = request.form['pendidikan']
        pekerjaan_ortu = request.form['pekerjaan_ortu']
        penghasilan_ortu = int(request.form['penghasilan_ortu'])
        model_type = request.form['model_type']

        input_df = pd.DataFrame([{
            'usia': usia,
            'jenis_kelamin': jenis_kelamin,
            'pendidikan': pendidikan,
            'pekerjaan_ortu': pekerjaan_ortu,
            'penghasilan_ortu': penghasilan_ortu
        }])

        if model_type == 'regresi':
            pred = reg_model.predict(input_df)[0]
            prediction = f"Estimasi biaya: Rp {int(pred):,}".replace(",", ".")
            model_used = "Regresi Linear"
        elif model_type == 'klasifikasi':
            pred = clf_model.predict(input_df)[0]
            prediction = f"Kategori biaya pendidikan: {pred}"
            model_used = "Klasifikasi"
        elif model_type == 'clustering':
            transformed = preprocessor_clust.transform(input_df)
            pred = clust_model.predict(transformed)[0]
            prediction = f"Masuk dalam kelompok pendidikan #{pred}"
            model_used = "Clustering"

    return render_template('index.html', prediction=prediction, model_used=model_used)

if __name__ == '__main__':
    app.run(debug=True)
