import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Mengimpor data
# Anambas AVG
url = 'https://raw.githubusercontent.com/ervanervan/dataset-skripsi/main/laporan_iklim_anambas_ff_avg_1.csv'
data = pd.read_csv(url)

# Mengonversi kolom 'Tanggal' ke tipe datetime
data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')

# Mengatur 'Tanggal' sebagai indeks
data.set_index('Tanggal', inplace=True)

# Menggunakan MinMaxScaler untuk menskalakan data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Modifikasi fungsi create_dataset
def create_dataset(dataset, timeseries=1):
    X, Y = [], []
    for i in range(len(dataset)-timeseries):
        end_ix = i + timeseries
        a = dataset[i:end_ix, 0]
        b = dataset[end_ix, 0]
        X.append(a)
        Y.append(b)
    return np.array(X), np.array(Y)

timeseries = 5
X, Y = create_dataset(data_scaled, timeseries)
X = np.reshape(X, (X.shape[0], timeseries, 1))

# Membagi data menjadi 70% training dan 30% testing
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

class PerformanceHistory(Callback):
    def __init__(self):
        self.rmse_train = []
        self.rmse_val = []
        self.mape_train = []
        self.mape_val = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Menghitung RMSE dan MAPE dari loss dan mae
        train_rmse = np.sqrt(logs['loss'])
        val_rmse = np.sqrt(logs['val_loss'])
        train_mape = logs['mae'] * 100
        val_mape = logs['val_mae'] * 100
        
        # Menyimpan nilai RMSE dan MAPE di list
        self.rmse_train.append(train_rmse)
        self.rmse_val.append(val_rmse)
        self.mape_train.append(train_mape)
        self.mape_val.append(val_mape)

# Modifikasi arsitektur model
def createModel():
    model = Sequential()
    model.add(Bidirectional(GRU(75, activation='tanh', return_sequences=False), input_shape=(timeseries, 1)))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

model = createModel()

# Melatih model
def trainingModel(model):
    callback_performance = PerformanceHistory()
    history = model.fit(X_train, Y_train, epochs=80, batch_size=64, validation_split=0.2, callbacks=[callback_performance])
    return history, callback_performance

history, callback_performance = trainingModel(model)

model_name = "Bidirectional_GRU_FF_AVG_ANAMBAS1"

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig("loss_plot_"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()

# Evaluasi model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and actual values to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
actual_train = scaler.inverse_transform(Y_train)
actual_test = scaler.inverse_transform(Y_test)

# Fungsi untuk menghitung metrik evaluasi
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy = 100 - mape
    return mse, rmse, mae, mape, accuracy

# Kalkulasi metrik untuk data pelatihan
mse_train, rmse_train, mae_train, mape_train, akurasi_train = calculate_metrics(actual_train, train_predict)
print('\nTraining Data:')
print(f'Train MSE: {mse_train:.4f}')
print(f'Train RMSE: {rmse_train:.4f}')
print(f'Train MAE: {mae_train:.4f}')
print(f'Train MAPE: {mape_train:.2f}%')
print(f'Train Accuracy: {akurasi_train:.2f}%')

# Kalkulasi metrik untuk data pengujian
mse_test, rmse_test, mae_test, mape_test, akurasi_test = calculate_metrics(actual_test, test_predict)
print('\nTesting Data:')
print(f'Test MSE: {mse_test:.4f}')
print(f'Test RMSE: {rmse_test:.4f}')
print(f'Test MAE: {mae_test:.4f}')
print(f'Test MAPE: {mape_test:.2f}%')
print(f'Test Accuracy: {akurasi_test:.2f}%')
print('\n')

BiGRU_Layer = model.layers[0]
BiGRU_Weight = BiGRU_Layer.get_weights()
input_weight = BiGRU_Weight[0][0]
bias = BiGRU_Weight[2][0]

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# Membuat dictionary untuk menyimpan kinerja model
model_performance = {
    'Training Performance': {
        'MSE': float(mse_train),
        'RMSE': float(rmse_train),
        'MAE': float(mae_train),
        'MAPE': float(mape_train),
        'Accuracy': float(akurasi_train)
    },
    'Testing Performance': {
        'MSE': float(mse_test),
        'RMSE': float(rmse_test),
        'MAE': float(mae_test),
        'MAPE': float(mape_test),
        'Accuracy': float(akurasi_test)
    },
    'Model Name': model_name,
    'Bobot': list(map(float, input_weight)),
    'Bias': list(map(float, bias))
}

# Menyimpan kinerja model ke dalam file JSON
with open(model_name + '.json', 'w') as json_file:
    json.dump(model_performance, json_file, indent=4, default=convert_to_serializable)

model_filename = "BiGRUFFAVGANB1.keras"
model.save(model_filename)

# Plot RMSE dan MAPE
plt.figure(figsize=(14, 7))

# Plotting RMSE
plt.subplot(1, 2, 1)
plt.plot(callback_performance.rmse_train, label='Training RMSE')
plt.plot(callback_performance.rmse_val, label='Validation RMSE')
plt.title('Training and Validation RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()

# Plotting MAPE
plt.subplot(1, 2, 2)
plt.plot(callback_performance.mape_train, label='Training MAPE')
plt.plot(callback_performance.mape_val, label='Validation MAPE')
plt.title('Training and Validation MAPE')
plt.xlabel('Epoch')
plt.ylabel('MAPE (%)')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data.index[:train_size], actual_train.flatten(), label='Data Aktual - Training')
plt.plot(data.index[:train_size], train_predict.flatten(), label='Prediksi - Training')
plt.plot(data.index[train_size:train_size+len(actual_test)], actual_test.flatten(), label='Data Aktual - Testing')
plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Prediksi - Testing')
plt.xlabel('Tanggal')
plt.ylabel('Kecepatan Angin Rata-Rata')
plt.title('Prediksi vs Data Aktual Kecepatan Angin Rata-Rata')
plt.legend()
plt.savefig("hasil_prediksi_90days_"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()

def predict_wind_speed_90days(model, scaler, input_data):
    input_data_reshaped = np.array(input_data).reshape(timeseries, 1)
    input_scaled = scaler.transform(input_data_reshaped)
    input_reshaped = np.reshape(input_scaled, (1, timeseries, 1))
    prediction_scaled = model.predict(input_reshaped)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0, 0]

inputan_kecepatan = scaler.inverse_transform(Y_test[-(timeseries):])
kecepatan_sebelumnya = [inputan_kecepatan[i][0] for i in range(len(inputan_kecepatan))]

forecasted = []
for i in range(90):
    input_x = np.array(kecepatan_sebelumnya).reshape(-1, 1)
    prediction = predict_wind_speed_90days(model, scaler, input_x)
    forecasted.append(prediction)
    kecepatan_sebelumnya.pop(0)
    kecepatan_sebelumnya.append(prediction)

plt.figure(figsize=(10, 12))
plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Data Predicition - Testing')
forecasted_dates = pd.date_range(start=data.index[train_size+len(test_predict)], periods=len(forecasted))
plt.plot(forecasted_dates, forecasted, label='Prediksi Kecepatan Angin')
plt.xlabel('Tanggal')
plt.ylabel('Kecepatan Angin')
plt.title('Prediksi Kecepatan Angin Rata-Rata Selama 90 Hari')
plt.legend()
plt.grid(True)
plt.savefig("forecasting_"+model_name+'.jpeg', format='jpeg', dpi=1000)
plt.show()
