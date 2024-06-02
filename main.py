# from gruModelTgpinangFFAVG import predict_ff_avg_Tgpinang,calculate_prediction_accuracy,df
from gru_anambas_ff_x import predict_ff_x_anb, data_ff_x_anb, X_data_ff_x_anb, predict_original_data_ff_x_anb, predict_multiple_day_ff_x_anb
from gru_anambas_ff_avg import predict_ff_avg_anb, data_ff_avg_anb, X_data_ff_avg_anb, predict_original_data_ff_avg_anb, predict_multiple_day_ff_avg_anb
from gru_tanjungpinang_ff_avg import predict_ff_avg_tgpinang, data_ff_avg_tgpinang, X_data_ff_avg_tgpinang, predict_original_data_ff_avg_tgpinang, predict_multiple_day_ff_avg_tgpinang
from gru_tanjungpinang_ff_x import predict_ff_x_tgpinang, data_ff_x_tgpinang, X_data_ff_x_tgpinang, predict_original_data_ff_x_tgpinang, predict_multiple_day_ff_x_tgpinang

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
    forecasted = predict_multiple_day_ff_avg_tgpinang(day=7)
    if request.method == 'POST':
        input_kecepatan_sebelumnya = str(request.form["input"])
        # Memisahkan string berdasarkan koma
        string_list = input_kecepatan_sebelumnya.split(',')

        int_list = [int(i) for i in string_list]
        input_kecepatan_sebelumnya = int_list
        prediction = predict_ff_avg_tgpinang(input_kecepatan_sebelumnya)
        
        # Lakukan apa pun yang diperlukan dengan input_date di sini
        prediksi = prediction

        return render_template('tgpinang_ff_avg.html', prediksi=round(prediksi,3),forecasting=forecasted)
    if request.method == 'GET':
        return render_template('tgpinang_ff_avg.html', prediksi=prediksi, forecasting=forecasted)
    
    
@app.route('/tanjungpinang/kecepatan_angin_rata_rata/dataset', methods=['GET'])
def tgpinang_ff_avg_dataset():
    predictions_data = predict_original_data_ff_avg_tgpinang(X_data_ff_avg_tgpinang)
    
    if request.method == 'GET':
        return render_template('tgpinang_ff_avg_dataset.html', data=data_ff_avg_tgpinang, predictions=predictions_data)
    

@app.route('/tanjungpinang/kecepatan_angin_rata_rata/model', methods=['GET'])
def tgpinang_ff_avg_model():

    if request.method == 'GET':
        filename = 'GRU_FF_AVG_TGPINANG.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('tgpinang_ff_avg_model.html', mape_train=round(performance_data['Training Performance']['MAPE'],3), rmse_train=round(performance_data['Training Performance']['RMSE'],3), akurasi_train=round(performance_data['Training Performance']['Accuracy'],3), mape_test=round(performance_data['Testing Performance']['MAPE'],3), rmse_test=round(performance_data['Testing Performance']['RMSE'],3), akurasi_test=round(performance_data['Testing Performance']['Accuracy'],3))
    
    
@app.route('/tanjungpinang/kecepatan_angin_rata_rata/bobot', methods=['GET'])
def tgpinang_ff_avg_bobot():
    if request.method == 'GET':
        filename = 'GRU_FF_AVG_TGPINANG.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('tgpinang_ff_avg_bobot.html', bobot=performance_data['Bobot'])
    

@app.route('/tanjungpinang/kecepatan_angin_rata_rata/bias', methods=['GET'])
def tgpinang_ff_avg_bias():

    if request.method == 'GET':
        filename = 'GRU_FF_AVG_TGPINANG.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('tgpinang_ff_avg_bias.html', bias=performance_data['Bias'])
    

# Tanjungpinang Kecepatan angin maksimum
@app.route('/tanjungpinang/kecepatan_angin_maksimum', methods=['GET', 'POST'])
def tgpinang_ff_x():
    prediksi = False
    forecasted = predict_multiple_day_ff_x_tgpinang(day=7)
    if request.method == 'POST':
        input_kecepatan_sebelumnya = str(request.form["input"])
        # Memisahkan string berdasarkan koma
        string_list = input_kecepatan_sebelumnya.split(',')

        int_list = [int(i) for i in string_list]
        input_kecepatan_sebelumnya = int_list
        prediction = predict_ff_x_tgpinang(input_kecepatan_sebelumnya)
        
        # Lakukan apa pun yang diperlukan dengan input_date di sini
        prediksi = prediction

        return render_template('tgpinang_ff_x.html', prediksi=round(prediksi,3),forecasting=forecasted)
    if request.method == 'GET':
        return render_template('tgpinang_ff_x.html', prediksi=prediksi, forecasting=forecasted)


@app.route('/tanjungpinang/kecepatan_angin_maksimum/dataset', methods=['GET'])
def tgpinang_ff_x_dataset():
    
    predictions_data = predict_original_data_ff_x_tgpinang(X_data_ff_x_tgpinang)
    
    if request.method == 'GET':
        return render_template('tgpinang_ff_x_dataset.html', data=data_ff_x_tgpinang, predictions=predictions_data)
    

@app.route('/tanjungpinang/kecepatan_angin_maksimum/model', methods=['GET'])
def tgpinang_ff_x_model():

    if request.method == 'GET':
        filename = 'GRU_FF_X_TGPINANG.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('tgpinang_ff_x_model.html', mape_train=round(performance_data['Training Performance']['MAPE'],3), rmse_train=round(performance_data['Training Performance']['RMSE'],3), akurasi_train=round(performance_data['Training Performance']['Accuracy'],3), mape_test=round(performance_data['Testing Performance']['MAPE'],3), rmse_test=round(performance_data['Testing Performance']['RMSE'],3), akurasi_test=round(performance_data['Testing Performance']['Accuracy'],3))

@app.route('/tanjungpinang/kecepatan_angin_maksimum/bobot', methods=['GET'])
def tgpinang_ff_x_bobot():
    if request.method == 'GET':
        filename = 'GRU_FF_X_TGPINANG.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('tgpinang_ff_x_bobot.html', bobot=performance_data['Bobot'])
    

@app.route('/tanjungpinang/kecepatan_angin_maksimum/bias', methods=['GET'])
def tgpinang_ff_x_bias():

    if request.method == 'GET':
        filename = 'GRU_FF_X_TGPINANG.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('tgpinang_ff_x_bias.html', bias=performance_data['Bias'])
    

# Anambas Kecepatan angin rata-rata
@app.route('/anambas/kecepatan_angin_rata_rata', methods=['GET', 'POST'])
def anambas_ff_avg():
    prediksi = False
    forecasted = predict_multiple_day_ff_avg_anb(day=7)
    if request.method == 'POST':
        input_kecepatan_sebelumnya = str(request.form["input"])
        # Memisahkan string berdasarkan koma
        string_list = input_kecepatan_sebelumnya.split(',')

        # Mengonversi setiap elemen string menjadi integer
        int_list = [int(i) for i in string_list]
        input_kecepatan_sebelumnya = int_list
        prediction = predict_ff_avg_anb(input_kecepatan_sebelumnya)
        
        # Lakukan apa pun yang diperlukan dengan input_date di sini
        prediksi = prediction

        return render_template('anambas_ff_avg.html', prediksi=round(prediksi,3),forecasting=forecasted)
    if request.method == 'GET':
        
        return render_template('anambas_ff_avg.html', prediksi=prediksi, forecasting=forecasted)


@app.route('/anambas/kecepatan_angin_rata_rata/dataset', methods=['GET'])
def anambas_ff_avg_dataset():
    
    predictions_data = predict_original_data_ff_avg_anb(X_data_ff_avg_anb)
    
    if request.method == 'GET':
        return render_template('anambas_ff_avg_dataset.html', data=data_ff_avg_anb, predictions=predictions_data)
    

@app.route('/anambas/kecepatan_angin_rata_rata/model', methods=['GET'])
def anambas_ff_avg_model():

    if request.method == 'GET':
        filename = 'GRU_FF_AVG_ANAMBAS_NEW.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('anambas_ff_avg_model.html', mape_train=round(performance_data['Training Performance']['MAPE'],3), rmse_train=round(performance_data['Training Performance']['RMSE'],3), akurasi_train=round(performance_data['Training Performance']['Accuracy'],3), mape_test=round(performance_data['Testing Performance']['MAPE'],3), rmse_test=round(performance_data['Testing Performance']['RMSE'],3), akurasi_test=round(performance_data['Testing Performance']['Accuracy'],3))
    
    
@app.route('/anambas/kecepatan_angin_rata_rata/bobot', methods=['GET'])
def anambas_ff_avg_bobot():
    if request.method == 'GET':
        filename = 'GRU_FF_AVG_ANAMBAS_NEW.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('anambas_ff_avg_bobot.html', bobot=performance_data['Bobot'])

@app.route('/anambas/kecepatan_angin_rata_rata/bias', methods=['GET'])
def anambas_ff_avg_bias():

    if request.method == 'GET':
        filename = 'GRU_FF_AVG_ANAMBAS_NEW.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('anambas_ff_avg_bias.html', bias=performance_data['Bias'])
    
    
# Anambas Kecepatan angin maksimum
@app.route('/anambas/kecepatan_angin_maksimum', methods=['GET', 'POST'])
def anambas_ff_x():
    prediksi = False
    forecasted = predict_multiple_day_ff_x_anb(day=7)
    if request.method == 'POST':
        input_kecepatan_sebelumnya = str(request.form["input"])
        # Memisahkan string berdasarkan koma
        string_list = input_kecepatan_sebelumnya.split(',')

        # Mengonversi setiap elemen string menjadi integer
        int_list = [int(i) for i in string_list]
        input_kecepatan_sebelumnya = int_list
        prediction = predict_ff_x_anb(input_kecepatan_sebelumnya)
        
        # Lakukan apa pun yang diperlukan dengan input_date di sini
        prediksi = prediction

        return render_template('anambas_ff_x.html', prediksi=round(prediksi,3),forecasting=forecasted)
    if request.method == 'GET':
        
        return render_template('anambas_ff_x.html', prediksi=prediksi, forecasting=forecasted)


@app.route('/anambas/kecepatan_angin_maksimum/dataset', methods=['GET'])
def anambas_ff_x_dataset():
    
    predictions_data = predict_original_data_ff_x_anb(X_data_ff_x_anb)
    
    if request.method == 'GET':
        return render_template('anambas_ff_x_dataset.html', data=data_ff_x_anb, predictions=predictions_data)
    

@app.route('/anambas/kecepatan_angin_maksimum/model', methods=['GET'])
def anambas_ff_x_model():
    if request.method == 'GET':
        filename = 'GRU_FF_X_ANAMBAS.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('anambas_ff_x_model.html', mape_train=round(performance_data['Training Performance']['MAPE'],3), rmse_train=round(performance_data['Training Performance']['RMSE'],3), akurasi_train=round(performance_data['Training Performance']['Accuracy'],3), mape_test=round(performance_data['Testing Performance']['MAPE'],3), rmse_test=round(performance_data['Testing Performance']['RMSE'],3), akurasi_test=round(performance_data['Testing Performance']['Accuracy'],3))
    
    
@app.route('/anambas/kecepatan_angin_maksimum/bobot', methods=['GET'])
def anambas_ff_x_bobot():
    if request.method == 'GET':
        filename = 'GRU_FF_X_ANAMBAS.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('anambas_ff_x_bobot.html', bobot=performance_data['Bobot'])

@app.route('/anambas/kecepatan_angin_maksimum/bias', methods=['GET'])
def anambas_ff_x_bias():

    if request.method == 'GET':
        filename = 'GRU_FF_X_ANAMBAS.json'
    
        with open(filename, 'r') as file :
            performance_data = json.load(file)
        
        return render_template('anambas_ff_x_bias.html', bias=performance_data['Bias'])
    
app.run(debug=True)
