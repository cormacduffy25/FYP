from flask import render_template, Blueprint, request, jsonify
from flask import Flask
from sqlalchemy import create_engine, insert, MetaData, Table
import pandas as pd
import json


def get_db_url():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config['db_url']

db_url = get_db_url()
engine = create_engine(db_url)
predictions_ann = Table('predictions_ann', MetaData(), autoload_with=engine)


main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/api/data/ann', methods=['GET'])
def get_data_ann():
    df = pd.read_sql_query('SELECT * FROM predictions_ann ORDER BY year ASC', engine)

    response_data = {
        'labels': df['year'].tolist(),
        'datasets': [
            {
                'label': 'Actual',
                'data': df['actual'].tolist(),
                'fill': False,
                'borderColor': '#FF6384',
                'backgroundColor': '#FF6384'
            },
            {
                'label': 'Prediction',
                'data': df['prediction'].tolist(),
                'fill': False,
                'borderColor': '#36A2EB',
                'backgroundColor': '#36A2EB'
            }
        ]
}
    return jsonify(response_data)

@main_blueprint.route('/api/data/svm', methods=['GET'])
def get_data_svm():
    df = pd.read_sql_query('SELECT * FROM svm_prediction ORDER BY year ASC', engine)

    response_data = {
        'labels': df['year'].tolist(),
        'datasets': [
            {
                'label': 'Actual',
                'data': df['actual'].tolist(),
                'fill': False,
                'borderColor': '#FF6384',
                'backgroundColor': '#FF6384'
            },
            {
                'label': 'Prediction',
                'data': df['prediction'].tolist(),
                'fill': False,
                'borderColor': '#36A2EB',
                'backgroundColor': '#36A2EB'
            }
        ]
}
    return jsonify(response_data)

@main_blueprint.route('/')
def home():
    return render_template('home.html')

@main_blueprint.route('/about/')
def about():
    return render_template('about.html')

@main_blueprint.route('/contact/')
def contact():
    return render_template('contact.html')

@main_blueprint.route('/svm/')
def svm():
    return render_template('svm.html')

@main_blueprint.route('/ann/')
def ann():
    return render_template('ann.html')