import joblib
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify, abort
import random
import os
from playhouse.db_url import connect
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError, DateTimeField
)
from playhouse.shortcuts import model_to_dict
import sklearn
from datetime import datetime

#Database

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    proba = FloatField()
    outcome = TextField(null=False)
    predicted_outcome = TextField(null=False)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)


class Received(Model):
    date_id = DateTimeField(default=datetime.now)
    observation = TextField()

    class Meta:
        database = DB

DB.create_tables([Received], safe=True)


# Reading Latitude/Longitude dictionaries

with open(os.path.join("data", "lat_dict.json")) as json_file:

    lat_dict = json.load(json_file)

with open(os.path.join("data", "long_dict.json")) as json_file:
    long_dict = json.load(json_file)  


#Unpickling

with open('columns.json') as fh:

    columns = json.load(fh)
    

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')

# Server

app = Flask(__name__)

@app.route('/should_search/', methods = ['POST'])
def should_search():
    
    payload = request.get_json()
    response = {}
    
    r = Received(observation = payload)

    try:
        r.save()
    except:
        pass

    try:
        payload['observation_id']
        print(payload['observation_id'])
    except:
        # response = {'observation_id': None, 
        #             'error': 'Observation_id is missing.'}
        # return jsonify(response)
        abort(405, description='Mising observation_id')
    
    observation_id = payload['observation_id']
    observation = payload
    
    if len(observation) < 12: 
        # response = {'observation_id': observation_id,
        #             'error': 'Observation data has extra information, please revise.'}
        # return jsonify(response)
        abort(405, description='Incorrect or missing data.')
        
    
    if len(observation) > 12:
        # response = {'observation_id': observation_id,
        #             'error': 'Observation data has extra information, please revise.'}
        # return jsonify(response)
        abort(405, description='Incorrect or missing data.')
    
    age_options = ['25-34', 'over 34', '10-17', '18-24', 'under 10']
    if observation['Age range'] not in age_options:
        # response = {'observation_id': observation_id,
        #             'error': 'Age range is invalid.'}
        # return jsonify(response)
        abort(405, description='Invalid age range.')

    if observation['Gender'] not in ['Male','Female','Other']:
        # response = {'observation_id': observation_id,
        #             'error': 'Please revise the Gender data.'}
        # return jsonify(response)
        abort(405, description='Invalid gender data.')


    if observation['Type'] not in ['Person search', 'Person and Vehicle search', 'Vehicle search']:
        # response = {'observation_id': observation_id,
        #             'error': 'Type of search is invalid.'}
        # return jsonify(response)
        abort(405, description='Invalid type of search data.')
    
    
    try:
        pd.to_datetime(observation['Date'],infer_datetime_format=True)
    except:
        # response = {'observation_id': None, 
        #             'error': 'Date field is invalid.'}
        # return jsonify(response)
        abort(405, description='Invalid date data.')
    
    if observation['Officer-defined ethnicity'] not in ['White', 'Other', 'Asian', 'Black', 'Mixed']:
        # response = {'observation_id': observation_id,
        #             'error': 'Please revise the ethnicity data to one of the possible options (White, Other, Asian, Black, Mixed).'}
        # return jsonify(response)
        abort(405, description='Invalid ethnicity data.')

    try:
        float(observation['Longitude'])
    except:
        # response = {'observation_id': None, 
        #             'error': 'something is wrong with the Longitude, please revise'}
        # return jsonify(response)
        abort(405, description='Invalid longitude data.')
    
    try:
        float(observation['Latitude'])
    except:
        # response = {'observation_id': None, 
        #             'error': 'something is wrong with the Latitude, please revise'}
        # return jsonify(response)
        abort(405, description='Invalid latitude data.')
    

    obs = pd.DataFrame([payload], columns = columns).astype(dtypes)

    proba = pipeline.predict_proba(obs)[0,1]
    #proba = random.randint(0,1)
    outcome = 'False'
    if proba >= 0.5:
        outcome = 'True'
    
    response = {
        'observation_id': observation_id,
        'outcome': outcome
    }
    
    p = Prediction(observation_id = observation_id,
                   proba = proba,
                   outcome = outcome,
                   predicted_outcome = outcome,
                   observation = observation)
    
   

    try:
        p.save()
    except IntegrityError as e:

        error_msg = "Observation ID: '{}' already exists".format(observation_id)
        response["error"] = error_msg
        response["outcome"] = outcome
        #print(error_msg)
        DB.rollback()
    
    return jsonify(response)


@app.route('/search_result/', methods = ['POST'])
def search_result():
    payload = request.get_json()

    r = Received(observation = payload)

    try:
        r.save()
    except:
        pass

    try:
        p = Prediction.get(Prediction.observation_id == payload['observation_id'])
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

# if __name__ == "__main__":
#     app.run(debug=True)

