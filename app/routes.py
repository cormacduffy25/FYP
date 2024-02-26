from flask import render_template, Blueprint, request
from flask import Flask

main_blueprint = Blueprint('main', __name__)

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