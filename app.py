import os
import time
import json
import torch
import warnings

warnings.filterwarnings("ignore")

from threading import Thread
from datetime import datetime
from flask import redirect, url_for
from flask import Flask, render_template, request, jsonify, session
from prompts import load_prompts
from model import model_routes


app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key in production

app.register_blueprint(model_routes)

typing_metrics_id = 0
typing_metrics_dictionary = {}
prompts = load_prompts()


@app.route('/')
def index():
    global typing_metrics_id

    if not prompts:
        return "No prompts available. Please add prompts to prompts.txt."
    
    typing_metrics_id += 1
    return render_template('index.html', prompt_id=prompts[0]['id'], total_prompts=len(prompts), typing_metrics_id=typing_metrics_id)
    

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()

    current_typing_metrics_id = data['typing_metrics_id']
    prompt_id = data['prompt_id']
    next_prompt_id = None
    for i, prompt in enumerate(prompts):
        if prompt['id'] == prompt_id:
            next_prompt_id = prompts[i + 1]['id'] if i + 1 < len(prompts) else None
            break 

    overall_metrics = data['overall_metrics']
    if current_typing_metrics_id not in typing_metrics_dictionary:
        typing_metrics_dictionary[current_typing_metrics_id] = {
            'session_start_time': overall_metrics['start_time'],
            'session_end_time': overall_metrics['end_time'],
            'prompt_responses': {},
            'total_prompts': 0
        }
    typing_metrics_dictionary[current_typing_metrics_id]['session_end_time'] = overall_metrics['end_time']
    typing_metrics_dictionary[current_typing_metrics_id]['total_prompts'] += 1

    prompt_responses = typing_metrics_dictionary[current_typing_metrics_id]['prompt_responses']
    prompt_responses[prompt_id] = {
        'prompt': data['prompt'],
        'response': data['response'],
        'word_metrics': data['word_metrics'],
        'overall_metrics': data['overall_metrics']
    }

    if next_prompt_id is None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'typing_metrics_{timestamp}_{current_typing_metrics_id}.json'
        filepath = os.path.join('typing_data', filename)
        with open(filepath, 'w') as f:
            json.dump(typing_metrics_dictionary[current_typing_metrics_id], f, indent=4)

        return jsonify({'status': 'finished'}), 200
    return jsonify({
        'status': 'success',
        'next_prompt_id': next_prompt_id
    }), 200


if __name__ == '__main__':
    app.run(debug=True)
