from flask import Flask, url_for, request, send_from_directory
from flask import render_template, redirect
from werkzeug.utils import secure_filename
import ensembles
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import os 
import matplotlib.pyplot as plt
from time import time


plt.style.use('ggplot')

app = Flask(__name__, template_folder='html')
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
ALLOWED_EXTENSIONS = set(['csv'])

datasets = {}
data_info = {}
models = {}
params = {}
message = ""
message_use = ""
train_RMSE = 0
train_time = 0
test_RMSE = 0
loss_fname = ""
preds = False


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/index_js')
def get_index():
    return '<html><center><script>document.write("hello from flask server")</script></center> </html>'


@app.route('/clear_graphics', methods=['POST'])
def clear_graphics():
    global train_RMSE
    train_RMSE = 0
    if os.path.isfile('static/' + loss_fname):
            os.remove('static/' + loss_fname)
    return redirect(url_for('index'))


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
            request.form['data_target']
        except:
            data_name = request.form['data_name']
            train_file = request.files['data_train_fname']
            target_file = request.files['data_target_fname']
            try:
                train_fname = secure_filename(train_file.filename)
                target_fname = secure_filename(target_file.filename)

                X = pd.read_csv(app.config['UPLOAD_FOLDER'] + "/" + train_fname,
                        index_col=index_col)
                y = pd.read_csv(app.config['UPLOAD_FOLDER'] + "/" + target_fname,
                        index_col=index_col).iloc[:, 0]
                datasets[data_name] = (X, y)

                data_info[data_name] = {'train_addres':train_fname, 
                        'target_addres':target_fname, 'data_shape':X.shape}
                message = "Done"
            except Exception as e:
                message = "Wrong path"
        else:
            data_target = request.form['data_target']
            data_name = request.form['data_name']
            data_file = request.files['data_fname']
            if data_file:
                filename = secure_filename(data_file.filename)
                data_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                try:
                    data = pd.read_csv(app.config['UPLOAD_FOLDER'] + "/" + filename, index_col=index_col)
                    X = data.drop(columns=[data_target])
                    y = data[data_target].copy()
                    data = None
                    datasets[data_name] = (X, y)

                    data_info[data_name] = {'filename':filename, 
                            'data_target':data_target, 'data_shape':X.shape}
                    message = "Done"
                except Exception as e:
                    message = "Wrong path"
        return redirect(url_for('prepare_data'))
    return render_template('datasets.html', message=message, data_info=data_info)


@app.route('/models', methods=['GET', 'POST'])
def prepare_model():
    global train_RMSE, loss_fname, train_time
    if request.method == 'POST':
        # parse params
        try:
            n_estimators = int(request.form['n_estimators'])
        except:
            n_estimators = 100

        try:
            max_depth = int(request.form['max_depth'])
        except:
            if request.form['select_model'] == 'RandomForest':
                max_depth = None
            else:
                max_depth = 5

        try:
            feature_subsample_size = int(request.form['feature_subsample_size'])
        except:
            feature_subsample_size = None

        try:
            learning_rate = float(request.form['learning_rate'])
        except:
            learning_rate = 0.1

        params[request.form['model_name']] = {"n_estimators":n_estimators, 
                "max_depth":max_depth, "feature_subsample_size":feature_subsample_size,
                "dataset":request.form['select_dataset']}

        # create model
        if request.form['select_model'] == 'RandomForest':
            models[request.form['model_name']] = ensembles.RandomForestMSE(
                    n_estimators, max_depth=max_depth, feature_subsample_size=feature_subsample_size
                )

        else:
            models[request.form['model_name']] = ensembles.GradientBoostingMSE(
                    n_estimators, max_depth=max_depth, feature_subsample_size=feature_subsample_size,
                    learning_rate=learning_rate
                )
            params[request.form['model_name']]["learning_rate"] = learning_rate

        # fit model
        st = time()
        X, y = datasets[request.form['select_dataset']]
        models[request.form['model_name']].fit(X.values, y.values)
        train_time = time() - st
        # grafics
        model = models[request.form['model_name']]
        loss = []
        if os.path.isfile('static/' + loss_fname):
            os.remove('static/' + loss_fname)
        fig_loss = plt.figure(figsize=(6, 4))
        if request.form['select_model'] == 'RandomForest':
            plt.title("RandomForest, RMSE")
            rez = np.zeros(X.values.shape[0])
            for i in range(model.n_estimators):
                rez += model.forest[i].predict(X.values[:, model.feat_subsamps[i]])
                loss.append(mean_squared_error(rez / (i + 1), y.values, squared=False))
        else:
            plt.title("GradientBoosting, RMSE")
            rez = model.f_0*np.ones(X.values.shape[0])
            for i in range(model.M):
                a, coef = model.estimators[i][0]
                rez -= coef*model.l_r*a.predict(X.values[:, model.feat_subsamps[i]])
                loss.append(mean_squared_error(rez, y.values, squared=False))
        plt.plot(loss)
        plt.xlabel("n_estimators")
        plt.ylabel("RMSE")
        loss_plot = True
        loss_fname = "loss_" + request.form['model_name'] + str(np.random.randint(1, 1000)) + ".jpg"
        fig_loss.savefig('static/' + loss_fname)
        train_RMSE = mean_squared_error(models[request.form['model_name']].predict(X.values), y.values, squared=False)
        return redirect(url_for('prepare_model'))
    return render_template('models.html', datasets=datasets, test=round(train_RMSE, 2),
            time=round(train_time, 2), loss_fname=loss_fname)


@app.route('/model_use', methods=['GET', 'POST'])
def use_model():
    global test_RMSE, message_use, preds
    if request.method == 'POST':
        data_file = request.files['datatest']
        if data_file:
            filename = secure_filename(data_file.filename)
            data_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)
            try:
                request.form['index_col']
            except:
                index_col = None
            else:
                index_col = 0
            try:
                X = pd.read_csv(app.config['UPLOAD_FOLDER'] + "/" + filename, index_col=index_col)
                model = models[request.form['model_name']]
                preds = model.predict(X.values)
                preds = pd.Series(preds)
                preds.to_csv("preds.csv", index=True)
                path = os.path.join(app.root_path, "preds.csv")
                preds = True
                return redirect(url_for('download'))
                message_use = Done
            except:
                message_use = "Something went wrong"
        return redirect(url_for('use_model'))
    return render_template('model_use.html', models=models, message=message_use,
            params=params)

@app.route('/preds.csv')
def download():
    global preds
    if preds:
        preds = False
        return send_from_directory(app.root_path, "./preds.csv")
    return redirect(url_for('use_model'))
