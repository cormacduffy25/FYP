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

@main_blueprint.route('/api/costs', methods=['GET'])
def calculate_costs():
    model = request.args.get('model', 'ann')
    year = request.args.get('year', type=int)
    kwh = request.args.get('kwh', type=float)  # Ensure kwh is obtained as a float

    # Assuming `fetch_prediction_data` returns the price per kWh
    predicted_price = fetch_prediction_data(year, model)
    if predicted_price is None:
        return jsonify({'error': 'No data available for selected year and model'}), 404

    if kwh is None or kwh < 0:
        return jsonify({'error': 'Invalid kWh value. Please provide a positive number.'}), 400

    cost = (predicted_price * kwh)/100  # Calculate total cost based on kWh input and model prediction
    return jsonify({'cost': cost})

def fetch_prediction_data(year, model):
    table = 'predictions_ann' if model.lower() == 'ann' else 'svm_prediction'
    query = f"SELECT prediction FROM {table} WHERE year = {year}"
    df = pd.read_sql_query(query, engine)
    return df['prediction'].iloc[0] if not df.empty else None



@main_blueprint.route('/')
def home():
    return render_template('home.html')

@main_blueprint.route('/about/')
def about():
    return render_template('about.html')

@main_blueprint.route('/contact/')
def contact():
    return render_template('contact.html')

@main_blueprint.route('/models/')
def models():
    return render_template('models.html')