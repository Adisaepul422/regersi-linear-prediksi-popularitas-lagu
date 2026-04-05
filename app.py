from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os

app = Flask(__name__)

# 🔧 Ambil path folder project (biar Vercel bisa baca file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🧠 Training model
def train_model():
    csv_path = os.path.join(BASE_DIR, 'song_data.csv')
    df = pd.read_csv(csv_path)

    X = df[['loudness']]
    Y = df['song_popularity']

    model = LinearRegression()
    model.fit(X, Y)
    return model

# Load model saat pertama jalan
model = train_model()

# 🌐 Route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    loudness_input = None
    
    if request.method == 'POST':
        try:
            loudness_input = float(request.form['loudness'])
            pred = model.predict(np.array([[loudness_input]]))
            prediction = round(pred[0], 2)
        except:
            prediction = "Input tidak valid"
        
    return render_template('index.html', prediction=prediction, loudness=loudness_input)

# 🔥 WAJIB untuk Vercel
app = app

# Untuk local run
if __name__ == '__main__':
    app.run(debug=True)