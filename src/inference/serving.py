from flask import Flask, request, jsonify
import torch
from ..inference.recommender import Recommender

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    top_k = data.get('top_k', 10)
    
    recommendations = recommender.get_recommendations(
        user_id, top_k=top_k
    )
    
    return jsonify({
        'user_id': user_id,
        'recommendations': [
            {'item_id': item_id, 'score': float(score)}
            for item_id, score in recommendations
        ]
    })