from flask import Blueprint, render_template, request
from .bio_generator import generate_bio_with_langchain
from flask import jsonify


main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/generate', methods=['POST'])
def generate():
    preferences = {
        "career": request.form.get("career"),
        "personality": request.form.get("personality"),
        "interests": request.form.get("interests"),
        "relationship": request.form.get("relationship"),
    }
    bio = generate_bio_with_langchain(preferences)
    
    return jsonify({"bio": bio})

