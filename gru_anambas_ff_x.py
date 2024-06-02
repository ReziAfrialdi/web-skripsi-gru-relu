import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

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

# Mengimpor data
link_data_set = 'https://raw.githubusercontent.com/ReziAfrialdi/dataset-skripsi/main/laporan_iklim_anambas_ff_x.csv'
data_kecepatan_angin = pd.read_csv(link_data_set)
data_ff_x_anb = pd.read_csv(link_data_set)

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
# X= np.reshape(Y, (Y.shape[0], timeseries, 1))
X = np.reshape(X, (X.shape[0], timeseries, 1))
X_data_ff_x_anb = X
# Membagi data menjadi 70% training dan 30% testing
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Load model
model_filename='GRU_FF_X_ANAMBAS.keras'
model = tf.keras.models.load_model(model_filename)

def predict_ff_x_anb(input_data):
    # Menambah dimensi agar sesuai dengan format yang diharapkan oleh scaler.transform()
    input_data_reshaped = np.array(input_data).reshape(timeseries, 1)  # Ubah dimensi input menjadi (5, 1)
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


def predict_original_data_ff_x_anb(input_data) :
    predictions = model.predict(input_data)
    results = scaler.inverse_transform(predictions).flatten()
    data = []
    for i in range(len(results)):
        data.append(round(float(results[i]),3))
    for i in range(timeseries):
        data.insert(0,"-")
    return data

def predict_multiple_day_ff_x_anb(day):
    data_y= Y_test
    data_y_reshape = np.reshape(data_y, (data_y.shape[0], 1))
    y_forcasted = data_y_reshape
    # Contoh penggunaan: memprediksi kecepatan angin maksimum untuk hari berikutnya berdasarkan 5 hari sebelumnya
    inputan_kecepatan = scaler.inverse_transform(y_forcasted[-(timeseries):])
    kecepatan_sebelumnya = []
    for i in range (len(inputan_kecepatan)):
        kecepatan_sebelumnya.append(inputan_kecepatan[i][0])
    # print(kecepatan_sebelumnya)

    forecasted = []
    for i in range(day):
        input_x = np.array(kecepatan_sebelumnya).reshape(-1, 1)
        prediction = predict_ff_x_anb(input_x)
        forecasted.append(prediction)
        kecepatan_sebelumnya.pop(0)
        kecepatan_sebelumnya.append(prediction)

    return forecasted



# cara penggunaan =>  prediksi 1 hari
# data_input = float(input("masukkan kecepatan angin hari sebelumnya : "))
# print(predict_ff_x_anb(data_input))

# cara penggunaan prediksi 1 minggu kedepan dari data set
# forecasted = predict_multiple_day_ff_x_anb(day=7)
# print(forecasted)