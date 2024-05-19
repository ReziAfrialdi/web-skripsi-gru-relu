# from gruModelTgpinangFFAVG import predict_ff_avg_Tgpinang,calculate_prediction_accuracy,df
from gru_anambas_ff_x import predict_ff_x_anb, data_ff_x_anb
from gru_anambas_ff_avg import predict_ff_avg_anb, data_ff_avg_anb
from gru_tanjungpinang_ff_avg import predict_ff_avg_tgpinang, data_ff_avg_tgpinang
from gru_tanjungpinang_ff_x import predict_ff_x_tgpinang, data_ff_x_tgpinang

from flask import Flask
from flask import url_for
from flask import request
from flask import render_template
from datetime import datetime
import json
app=Flask(__name__)

def get_day_of_year(date_string):
    # Ubah string menjadi objek datetime
    date_object = datetime.strptime(date_string, '%d-%m-%Y')
    
    # Dapatkan hari dalam setahun (day of year)
    day_of_year = date_object.timetuple().tm_yday
    
    return day_of_year

@app.route('/')
def index():
    return render_template('index.html')

# Tanjungpinang Kecepatan angin rata-rata
@app.route('/tanjungpinang/kecepatan_angin_rata_rata', methods=['GET', 'POST'])
def tgpinang_ff_avg():
    prediksi = False
    if request.method == 'POST':
        input_kecepatan_sebelumnya = str(request.form["input"])

        prediction = predict_ff_avg_tgpinang(input_kecepatan_sebelumnya)
        
        # Lakukan apa pun yang diperlukan dengan input_date di sini
        prediksi = prediction

        return render_template('tgpinang_ff_avg.html', prediksi=round(prediksi,3))
    if request.method == 'GET':
        return render_template('tgpinang_ff_avg.html', prediksi=prediksi)


@app.route('/tanjungpinang/kecepatan_angin_rata_rata/dataset', methods=['GET'])
def tgpinang_ff_avg_dataset():
    
    
    if request.method == 'GET':
        return render_template('tgpinang_ff_avg_dataset.html', data=data_ff_avg_tgpinang)
    

@app.route('/tanjungpinang/kecepatan_angin_rata_rata/model', methods=['GET'])
def tgpinang_ff_avg_model():
    
    
    if request.method == 'GET':
        filename = 'GRU_FF_AVG_TGPINANG.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('tgpinang_ff_avg_model.html', mape=round(performance_data['MAPE'],3), rmse=round(performance_data['RMSE'],3), akurasi=round(performance_data['AKURASI'],3))

# Tanjungpinang Kecepatan angin maksimum
@app.route('/tanjungpinang/kecepatan_angin_maksimum', methods=['GET', 'POST'])
def tgpinang_ff_x():
    prediksi = False
    if request.method == 'POST':
        input_kecepatan_sebelumnya = str(request.form["input"])

        prediction = predict_ff_x_tgpinang(input_kecepatan_sebelumnya)
        
        # Lakukan apa pun yang diperlukan dengan input_date di sini
        prediksi = prediction

        return render_template('tgpinang_ff_x.html', prediksi=round(prediksi,3))
    if request.method == 'GET':
        return render_template('tgpinang_ff_x.html', prediksi=prediksi)


@app.route('/tanjungpinang/kecepatan_angin_maksimum/dataset', methods=['GET'])
def tgpinang_ff_x_dataset():
    
    
    if request.method == 'GET':
        return render_template('tgpinang_ff_x_dataset.html', data=data_ff_x_tgpinang)
    

@app.route('/tanjungpinang/kecepatan_angin_maksimum/model', methods=['GET'])
def tgpinang_ff_x_model():
    
    
    if request.method == 'GET':
        filename = 'GRU_FF_X_TGPINANG.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('tgpinang_ff_x_model.html', mape=round(performance_data['MAPE'],3), rmse=round(performance_data['RMSE'],3), akurasi=round(performance_data['AKURASI'],3))

# Anambas Kecepatan angin rata-rata
@app.route('/anambas/kecepatan_angin_rata_rata', methods=['GET', 'POST'])
def anambas_ff_avg():
    prediksi = False
    if request.method == 'POST':
        input_kecepatan_sebelumnya = str(request.form["input"])

        prediction = predict_ff_avg_anb(input_kecepatan_sebelumnya)
        
        # Lakukan apa pun yang diperlukan dengan input_date di sini
        prediksi = prediction

        return render_template('anambas_ff_avg.html', prediksi=round(prediksi,3))
    if request.method == 'GET':
        return render_template('anambas_ff_avg.html', prediksi=prediksi)


@app.route('/anambas/kecepatan_angin_rata_rata/dataset', methods=['GET'])
def anambas_ff_avg_dataset():
    
    
    if request.method == 'GET':
        return render_template('anambas_ff_avg_dataset.html', data=data_ff_avg_anb)
    

@app.route('/anambas/kecepatan_angin_rata_rata/model', methods=['GET'])
def anambas_ff_avg_model():
    
    
    if request.method == 'GET':
        filename = 'GRU_FF_AVG_ANAMBAS.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('anambas_ff_avg_model.html', mape=round(performance_data['MAPE'],3), rmse=round(performance_data['RMSE'],3), akurasi=round(performance_data['AKURASI'],3))
    
# Anambas Kecepatan angin maksimum
@app.route('/anambas/kecepatan_angin_maksimum', methods=['GET', 'POST'])
def anambas_ff_x():
    prediksi = False
    if request.method == 'POST':
        input_kecepatan_sebelumnya = str(request.form["input"])

        prediction = predict_ff_x_anb(input_kecepatan_sebelumnya)
        
        # Lakukan apa pun yang diperlukan dengan input_date di sini
        prediksi = prediction

        return render_template('anambas_ff_x.html', prediksi=round(prediksi,3))
    if request.method == 'GET':
        return render_template('anambas_ff_x.html', prediksi=prediksi)


@app.route('/anambas/kecepatan_angin_maksimum/dataset', methods=['GET'])
def anambas_ff_x_dataset():
    
    
    if request.method == 'GET':
        return render_template('anambas_ff_x_dataset.html', data=data_ff_x_anb)
    

@app.route('/anambas/kecepatan_angin_maksimum/model', methods=['GET'])
def anambas_ff_x_model():
    
    
    if request.method == 'GET':
        filename = 'GRU_FF_X_ANAMBAS.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('anambas_ff_x_model.html', mape=round(performance_data['MAPE'],3), rmse=round(performance_data['RMSE'],3), akurasi=round(performance_data['AKURASI'],3))
    
    
app.run(debug=True)
