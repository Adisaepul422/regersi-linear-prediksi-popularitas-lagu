from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None  # model global

def train_model():
    try:
        csv_path = os.path.join(BASE_DIR, 'song_data.csv')
        df = pd.read_csv(csv_path)

        X = df[['loudness']]
        Y = df['song_popularity']

        model = LinearRegression()
        model.fit(X, Y)
        return model

    except Exception as e:
        print("ERROR TRAIN MODEL:", e)
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    global model

    prediction = None
    error = None


    if model is None:
        model = train_model()

    if request.method == 'POST':
        try:
            loudness_input = float(request.form['loudness'])

            if model is None:
                error = "Model gagal dimuat"
            else:
                pred = model.predict(np.array([[loudness_input]]))
                prediction = round(pred[0], 2)

        except Exception as e:
            error = f"Terjadi error: {e}"

    return render_template(
        'index.html',
        prediction=prediction,
        error=error
    )


app = app

if __name__ == '__main__':
    app.run(debug=True)