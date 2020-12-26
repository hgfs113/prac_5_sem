from flask import Flask, url_for, request
from flask import render_template, redirect
import ensembles
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import os 
import matplotlib.pyplot as plt
from time import time

plt.style.use('ggplot')

app = Flask(__name__, template_folder='html')


datasets = {}
data_info = {}
message = ""


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/index_js')
def get_index():
    return '<html><center><script>document.write("hello from flask server")</script></center> </html>'



@app.route('/datasets', methods=['GET', 'POST'])
def prepare_data():
    global message
    if request.method == 'POST':
        try:
            request.form['index_col']
        except:
            index_col = None
        else:
            index_col = 0

        try:
            request.form['data_train_fname']
        except:
            data_addres = request.form['data_fname']
            data_name = request.form['data_name']
            data_target = request.form['data_target']
            try:
                data = pd.read_csv(data_addres, index_col=index_col)
                X = data.drop(columns=[data_target])
                y = data[data_target].copy()
                data = None
                datasets[data_name] = (X, y)

                data_info[data_name] = {'data_addres':data_addres, 
                        'data_target':data_target, 'data_shape':X.shape}
                message = "Done"
            except Exception as e:
                message = "Wrong path"
        else:
            train_addres = request.form['data_train_fname']
            target_addres = request.form['data_target_fname']
            data_name = request.form['data_name']
            try:
                X = pd.read_csv(train_addres, index_col=index_col)
                y = pd.read_csv(target_addres, index_col=index_col).iloc[:, 0]
                datasets[data_name] = (X, y)

                data_info[data_name] = {'train_addres':train_addres, 
                        'target_addres':target_addres, 'data_shape':X.shape}
                message = "Done"
            except Exception as e:
                message = "Wrong path"

        return redirect(url_for('prepare_data'))
    return render_template('datasets.html', message=message, data_info=data_info)
