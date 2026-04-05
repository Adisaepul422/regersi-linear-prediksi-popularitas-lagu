from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Membangun model saat aplikasi pertama kali dijalankan
def train_model():
    df = pd.read_csv('song_data.csv')
    X = df[['loudness']]
    Y = df['song_popularity']
    model = LinearRegression()
    model.fit(X, Y)
    return model

model = train_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    loudness_input = None
    
    if request.method == 'POST':
        # Mengambil input dari form manual (Halaman 10 Modul)
        loudness_input = float(request.form['loudness'])
        # Melakukan prediksi
        pred = model.predict(np.array([[loudness_input]]))
        prediction = round(pred[0], 2)
        
    return render_template('index.html', prediction=prediction, loudness=loudness_input)

if __name__ == '__main__':
    app.run(debug=True)