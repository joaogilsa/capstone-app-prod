import joblib
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify
import random
import os
from playhouse.db_url import connect
import requests
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


#Database

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    proba = FloatField()
    outcome = TextField(null=True)
    predicted_outcome = TextField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)


#Unpickling

# with open('columns.json') as fh:
#     columns = json.load(fh)

# with open('dtypes.pickle', 'rb') as fh:
#     dtypes = pickle.load(fh)

# pipeline = joblib.load('pipeline.pickle')


# Server

app = Flask(__name__)

@app.route('/should_search', methods = ['POST'])
def should_search():
    payload = request.get_json()

    #backup request
    #url = os.environ.get('BACKUP_URL') #f"https://capstone-backup-production.up.railway.app/save_request"
    #requests.post(url, json=payload)

    observation_id = payload['observation_id']
    observation = payload
    # obs = pd.DataFrame([payload], columns = columns).asastype(dtypes)
    # proba = pipeline.predict(obs)[0,1]
    proba = random.randint(0,1)
    outcome = False
    if proba > 0.1:
        outcome = True
    
    response = {
        'observation_id': observation_id,
        'outcome': outcome
    }
    
    p = Prediction(observation_id = observation_id,
                   proba = proba,
                   outcome = outcome,
                   observation = observation)
    
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()

    return jsonify(response)


@app.route('/search_result', methods = ['POST'])
def search_result():
    payload = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == payload['observation_id'])
        p.predicted_outcome = p.outcome
        p.outcome = payload['outcome']
        p.save()
        response = {'observation_id':p.observation_id,
                    'outcome': p.outcome,
                    'predicted_outcome': p.predicted_outcome}
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(payload['observation_id'])
        return jsonify({'error': error_msg})
    
    # observation_id = payload['observation_id']
    # outcome = payload['outcome']
    # predicted_outcome = True
    # return jsonify({
    #     'observation_id': observation_id,
    #     'outcome': outcome,
    #     'predicted_outcome': predicted_outcome
    # })


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)



