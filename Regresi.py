import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Pembuatan Dataset (Contoh dari modul halaman 4)
data = {
    "Permintaan": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
    "Bahan_Baku": [110, 160, 210, 260, 310, 360, 420, 460, 510, 560, 620, 660, 710, 770, 810, 860, 920, 970, 1020, 1070]
}
df = pd.DataFrame(data)

# 2. Pemisahan Data
X = df[['Permintaan']]
Y = df['Bahan_Baku']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. Membangun Model
model = LinearRegression()
model.fit(X_train, Y_train)

# 4. Prediksi
Y_pred = model.predict(X_test)

# 5. Evaluasi
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared (R2): {r2:.2f}")

# 6. Visualisasi
plt.figure(figsize=(8, 5))
plt.scatter(X_test, Y_test, color='blue', label="Data Aktual")
plt.plot(X_test, Y_pred, color='red', label="Prediksi (Regresi Linear)")
plt.xlabel("Permintaan Produk")
plt.ylabel("Bahan Baku")
plt.title("Hasil Prediksi Regresi Linear")
plt.legend()
plt.show()