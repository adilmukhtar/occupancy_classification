import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
import random


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_test/', methods=['POST'])
def run_test():
    print('Test Started...')
    loaded_model = pickle.load(open('occupancy_model_rf.sav', 'rb'))
    df = pd.read_csv('datatest2.txt')
    df = np.asarray(df.drop(['date', 'Occupancy'], axis=1))
    print(df)

    occupied = loaded_model.predict(df[random.randint(0, len(df))].reshape(1,-1))
    print(occupied[0])
    data = {'occupied': str(occupied[0])}
    data = jsonify(data)
    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
