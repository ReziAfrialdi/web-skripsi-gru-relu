import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error

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


def createModel():
    model = Sequential()
    model.add(Input(shape=(timeseries,1)))
    model.add(GRU(30, activation='relu', return_sequences=False))
    model.add(Dense(1, activation='relu'))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def trainingModel(model):
    model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2)
    return model

def evaluate_model(actual_test,test_predict): 
    mse = mean_squared_error(actual_test, test_predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_test, test_predict)
    mape = np.mean(np.abs((actual_test - test_predict) / actual_test)) * 100
    akurasi = 100 - mape
    
    return mse, rmse, mae, mape, akurasi

def save_model_and_metrics(model, model_name, mse, rmse, mae, mape, akurasi):
    model_performance = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'AKURASI': akurasi,
        'model_name': model_name
    }
    
    with open(model_name + '.json', 'w') as json_file:
        json.dump(model_performance, json_file, indent=4)
    
    model_filename = model_name + ".keras"
    model.save(model_filename)


def predict_wind_speed_using_model(model, scaler, input_data):
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



def plot_data_aktual_data_prediksi(data_kecepatan_angin, actual_train, train_size, train_predict, actual_test, test_predict):
    # Gabungkan data aktual dan prediksi
    actual_data = np.concatenate((actual_train, actual_test), axis=0)
    predicted_data = np.concatenate((train_predict, test_predict), axis=0)
    
    # Plot data aktual vs prediksi
    plt.figure(figsize=(10,6))
    plt.plot(data_kecepatan_angin.index[:len(actual_data)], actual_data.flatten(), label='Data Aktual')
    plt.plot(data_kecepatan_angin.index[:len(predicted_data)], predicted_data.flatten(), label='Prediksi')
    plt.xlabel('Tanggal')
    plt.ylabel('Kecepatan Angin maksimum')
    plt.title('Prediksi vs Data Aktual Kecepatan Angin maksimum')
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
    plt.ylabel('Kecepatan Angin')
    plt.title('Prediksi Kecepatan Angin Maksimum Selama 7 Hari')

    # Tampilkan legenda dan grid
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_hasil_prediksi_kedepan_7_hari"+model_name+'.jpeg', format='jpeg', dpi=1000)

    # Tampilkan plot
    plt.show()


def predict_multiple_day(day, Y_test, timeseries,scaler, model):
    
    # Contoh penggunaan: memprediksi kecepatan angin maksimum untuk hari berikutnya berdasarkan 5 hari sebelumnya
    inputan_kecepatan = scaler.inverse_transform(Y_test[-(timeseries):])
    kecepatan_sebelumnya = []
    for i in range (len(inputan_kecepatan)):
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


# Mengimpor data
link_data_set = 'https://raw.githubusercontent.com/ReziAfrialdi/dataset-skripsi/main/laporan_iklim_harian_tanjungpinang_ff_x.csv'
data_kecepatan_angin = pd.read_csv(link_data_set)

# Mengonversi kolom 'Tanggal' ke tipe datetime
data_kecepatan_angin['Tanggal'] = pd.to_datetime(data_kecepatan_angin['Tanggal'], format='%d-%m-%Y')

# Mengatur 'Tanggal' sebagai indeks
data_kecepatan_angin.set_index('Tanggal', inplace=True)

# Menggunakan MinMaxScaler untuk menskalakan data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_kecepatan_angin)

timeseries = 1

# Mengubah bentuk input menjadi [samples, time steps, features]
X, Y = buat_dataset_timeseries(data_scaled, timeseries)
# X= np.reshape(Y, (Y.shape[0], timeseries, 1))
X = np.reshape(X, (X.shape[0], timeseries, 1))

# Membagi data menjadi 70% training dan 30% testing
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Modifikasi arsitektur model

# buat model
gru = createModel()

# Melatih model
model = trainingModel(gru)

# Evaluasi model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
#
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
actual_train = scaler.inverse_transform(Y_train)
actual_test = scaler.inverse_transform(Y_test)



mse, rmse, mae, mape, akurasi= evaluate_model(actual_test,test_predict)
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')
print(f'Akurasi: {akurasi:.2f}%')


model_name = "GRU_FF_X_TGPINANG"
save_model_and_metrics(model, model_name, mse, rmse, mae, mape, akurasi)

plot_data_aktual_data_prediksi(data_kecepatan_angin,actual_train,train_size,train_predict,actual_test,test_predict)

y_forcasted = Y_test
forecasted = predict_multiple_day(day=7, Y_test=y_forcasted, timeseries=timeseries,scaler=scaler, model=model)

plot_hasil_prediksi_kedepan(data_kecepatan_angin, forecasted, train_size,test_predict)