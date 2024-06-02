import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, GRU, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Untuk nyimpan rmse dan mape di setiap epoch epoch
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

# Modifikasi fungsi buat_dataset
def buat_dataset_timeseries(dataset, timeseries=1):
    X, Y = [], []
    for i in range(len(dataset)-timeseries):
        end_ix = i + timeseries
        a = dataset[i:end_ix, 0]
        b = dataset[end_ix, 0]
        X.append(a)
        Y.append(b)
    return np.array(X), np.array(Y)

# Buat model GRU
def createModel():
    model = Sequential()
    model.add(Input(shape=(timeseries,1)))
    model.add(GRU(30, activation='relu', return_sequences=False))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Untuk melatih model
def trainingModel(model):
    callback_performance = PerformanceHistory()
    model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2, callbacks=[callback_performance])
    return model, callback_performance

# Untuk mengevaluasi model
def evaluate_model(actual_test, test_predict): 
    mse = mean_squared_error(actual_test, test_predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_test, test_predict)
    mape = np.mean(np.abs((actual_test - test_predict) / actual_test)) * 100
    akurasi = 100 - mape
    
    return mse, rmse, mae, mape, akurasi

def convert_to_serializable(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
# Untuk menyimpan model dan metrik performa
def save_model_and_metrics(model, model_name, mse_train, rmse_train, mae_train, mape_train, akurasi_train, mse_test, rmse_test, mae_test, mape_test, akurasi_test):
        
    GRU_Layer = model.layers[0]
    GRU_Weight = GRU_Layer.get_weights()
    input_weight = GRU_Weight[0][0]
    bias = GRU_Weight[2][0]

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
    
    model_filename = model_name + ".keras"
    model.save(model_filename)

def predict_wind_speed_using_model(model, scaler, input_data):
    # Menambah dimensi agar sesuai dengan format yang diharapkan oleh scaler.transform()
    input_data_reshaped = np.array(input_data).reshape(timeseries, 1)  # Ubah dimensi input menjadi (timeseries, 1)
    # Menskalakan input data
    input_scaled = scaler.transform(input_data_reshaped)
    # Menambah dimensi untuk sesuai dengan bentuk input model
    input_reshaped = np.reshape(input_scaled, (1, timeseries, 1))
    # Melakukan prediksi
    # print(input_reshaped)
    prediction_scaled = model.predict(input_reshaped)
    # Membalikkan skala hasil prediksi ke skala aslinya
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0, 0]

def plot_data_aktual_data_prediksi(data_kecepatan_angin, actual_train, train_size, train_predict, actual_test, test_predict):
    plt.figure(figsize=(14, 7))

    # Plot training data
    plt.plot(data_kecepatan_angin.index[:train_size], actual_train, label='Data Aktual Training', color= '#9796FD')
    plt.plot(data_kecepatan_angin.index[:train_size], train_predict, label='Prediksi Training', color='#FC6565')

    # Plot testing data
    plt.plot(data_kecepatan_angin.index[train_size:train_size+len(actual_test)], actual_test, label='Data Aktual Testing', color='#E7A138')
    plt.plot(data_kecepatan_angin.index[train_size:train_size+len(test_predict)], test_predict, label='Prediksi Testing', color='#1A69E1')

    plt.xlabel('Tanggal')
    plt.ylabel('Kecepatan Angin Rata-rata (m/s)')
    plt.title('Data Aktual vs Prediksi Kecepatan Angin Rata-rata')
    plt.legend()
    plt.savefig("plot_data_aktual_data_prediksi_"+model_name+".jpeg", format='jpeg', dpi=1000)
    plt.show()

def plot_hasil_prediksi_kedepan(data_kecepatan_angin, forecasted, train_size, test_predict):
    # Plot data aktual vs prediksi
    plt.figure(figsize=(10,12))

    # Plot data aktual
    plt.plot(data_kecepatan_angin.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Data Predicition - Testing')

    # Plot prediksi kecepatan angin
    forecasted_dates = pd.date_range(start=data_kecepatan_angin.index[train_size+len(test_predict)], periods=len(forecasted))
    plt.plot(forecasted_dates, forecasted, label='Prediksi Kecepatan Angin')

    # Atur label dan judul plot
    plt.xlabel('Tanggal')
    plt.ylabel('Kecepatan Angin (m/s)')
    plt.title('Prediksi Kecepatan Angin Rata-rata Selama 7 Hari')

    # Tampilkan legenda dan grid
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_hasil_prediksi_kedepan_7_hari"+model_name+'.jpeg', format='jpeg', dpi=1000)

    # Tampilkan plot
    plt.show()

def predict_multiple_day(day, Y_test, timeseries, scaler, model):
    
    # Contoh penggunaan: memprediksi kecepatan angin rata-rata untuk hari berikutnya berdasarkan 5 hari sebelumnya
    inputan_kecepatan = scaler.inverse_transform(Y_test[-(timeseries):])
    kecepatan_sebelumnya = []
    for i in range(len(inputan_kecepatan)):
        kecepatan_sebelumnya.append(inputan_kecepatan[i][0])
    # print(kecepatan_sebelumnya)

    forecasted = []
    for i in range(day):
        input_x = np.array(kecepatan_sebelumnya).reshape(-1, 1)
        prediction = predict_wind_speed_using_model(model, scaler, input_x)
        forecasted.append(prediction)
        kecepatan_sebelumnya.pop(0)
        kecepatan_sebelumnya.append(prediction)

    return forecasted

# Grafik evaluasi data training dan testing - RMSE dan MAPE 
def plot_rmse_mape(callback_performance, model_name):
    # Plotting RMSE
    plt.subplot(1, 2, 1)
    plt.plot(callback_performance.rmse_train, label='Training RMSE')
    plt.plot(callback_performance.rmse_val, label='Testing RMSE')
    plt.title('Training and Testing RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()

    # Plotting MAPE
    plt.subplot(1, 2, 2)
    plt.plot(callback_performance.mape_train, label='Training MAPE')
    plt.plot(callback_performance.mape_val, label='Testing MAPE')
    plt.title('Training and Testing MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("plot_evaluasi_data_training_&_testing_RMSE_MAPE "+model_name+'.jpeg', format='jpeg', dpi=1000)
    plt.show()

# Mengimpor data
link_data_set = 'https://raw.githubusercontent.com/ReziAfrialdi/dataset-skripsi/main/laporan_harian_iklim_anambas_ff_avg_new.csv'
data_kecepatan_angin = pd.read_csv(link_data_set)

# Mengonversi kolom 'Tanggal' ke tipe datetime
data_kecepatan_angin['Tanggal'] = pd.to_datetime(data_kecepatan_angin['Tanggal'], format='%d-%m-%Y')

# Mengatur 'Tanggal' sebagai indeks
data_kecepatan_angin.set_index('Tanggal', inplace=True)

# Menggunakan MinMaxScaler untuk menskalakan data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_kecepatan_angin)

timeseries = 5

# Mengubah bentuk input menjadi [samples, time steps, features]
X, Y = buat_dataset_timeseries(data_scaled, timeseries)
X = np.reshape(X, (X.shape[0], timeseries, 1))

# Membagi data menjadi 70% training dan 30% testing
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# buat model
gru = createModel()

# Melatih model
model, callback_performance = trainingModel(gru)

# Evaluasi model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
actual_train = scaler.inverse_transform(Y_train)
actual_test = scaler.inverse_transform(Y_test)

# Evaluasi model untuk data training
mse_train, rmse_train, mae_train, mape_train, akurasi_train = evaluate_model(actual_train, train_predict)
print("\nEvaluasi hasil Data Training:")
print(f'MSE: {mse_train:.4f}')
print(f'RMSE: {rmse_train:.4f}')
print(f'MAE: {mae_train:.4f}')
print(f'MAPE: {mape_train:.2f}%')
print(f'Akurasi: {akurasi_train:.2f}%')

# Evaluasi model untuk data testing
mse_test, rmse_test, mae_test, mape_test, akurasi_test = evaluate_model(actual_test, test_predict)
print("\nEvaluasi hasil Data Testing:")
print(f'MSE: {mse_test:.4f}')
print(f'RMSE: {rmse_test:.4f}')
print(f'MAE: {mae_test:.4f}')
print(f'MAPE: {mape_test:.2f}%')
print(f'Akurasi: {akurasi_test:.2f}%')

model_name = "GRU_FF_AVG_ANAMBAS_NEW"

plot_rmse_mape(callback_performance, model_name)

save_model_and_metrics(model, model_name, mse_train, rmse_train, mae_train, mape_train, akurasi_train, mse_test, rmse_test, mae_test, mape_test, akurasi_test)

# Plot data aktual dan data prediksi untuk data training dan testing
plot_data_aktual_data_prediksi(data_kecepatan_angin, actual_train, train_size, train_predict, actual_test, test_predict)

# Prediksi kecepatan angin untuk 7 hari ke depan dan plot hasilnya
y_forecasted = Y_test
forecasted = predict_multiple_day(day=7, Y_test=y_forecasted, timeseries=timeseries, scaler=scaler, model=model)

plot_hasil_prediksi_kedepan(data_kecepatan_angin, forecasted, train_size, test_predict)