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
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key in production

# Directory to store typing metrics
DATA_DIR = 'typing_data'
# Path to the prompts file
PROMPTS_FILE = 'prompts.txt'

os.makedirs(DATA_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_responses = []
next_model_response_id = 0

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

def load_prompts():
    """Load prompts from the prompts.txt file with their unique IDs."""
    if not os.path.exists(PROMPTS_FILE):
        return []
    prompts = []
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line[0] == '#':
                continue # Skip comments starting with '#'
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue  # Skip malformed lines
            prompt_id, prompt_text = parts
            if len(prompt_id) != 5 or not prompt_id.isdigit():
                continue  # Ensure ID is 5-digit number
            prompts.append({'id': prompt_id, 'text': prompt_text})
    return prompts

@app.route('/')
def index():
    prompts = load_prompts()
    if not prompts:
        return "No prompts available. Please add prompts to prompts.txt."
    
    # Initialize session variables
    session['prompts'] = prompts
    session['current_prompt_index'] = 0
    session['responses'] = {}
    
    first_prompt = prompts[0]
    return render_template('index.html', prompt=first_prompt['text'], prompt_id=first_prompt['id'], total_prompts=len(prompts), current_prompt_number=1)

@app.route('/nextModelResponseId', methods=['GET'])
def get_next_model_response_id():
    global next_model_response_id
    next_model_response_id += 1
    return jsonify({'response_id': next_model_response_id})

@app.route('/getModelResponses', methods=['GET'])
def get_model_responses():
    return jsonify(model_responses)

@app.route('/getModelResponse/<int:response_id>', methods=['GET'])
def get_model_response(response_id):
    for model_response in model_responses:
        if model_response['response_id'] == response_id:
            return jsonify(model_response)
    return jsonify({'error': 'Model response not found.'}), 404

@app.route('/generateModelResponse', methods=['POST'])
def generate_model_response():
    data = request.get_json()
    response_id = data.get('response_id', 0)
    question = data.get('prompt', [])
    new_model_reponse = {
        'response_id': response_id,
        'question': question,
        'response': ''
    }

    model_responses.append(new_model_reponse)

    print(f"Generating response for response ID: {response_id}")
    print(f"Question: {question}")

    chat = [
        { "role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt")

    streamer = TextIteratorStreamer(tokenizer)

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=128)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for new_text in streamer:
        new_text = new_text.replace("\n", " ").replace("<eos>", "").replace("**", " ").replace("*", " ")
        if "<start_of_turn>model" in new_text:
            start_add_to_response = True
            continue
        if not start_add_to_response:
            continue
        new_model_reponse['response'] += new_text

    thread.join()

    return jsonify({'status': 'success', 'response_id': response_id}), 200
    

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    prompts = session.get('prompts', [])
    current_index = session.get('current_prompt_index', 0)
    
    if current_index >= len(prompts):
        return jsonify({'status': 'error', 'message': 'No more prompts available.'}), 400
    
    current_prompt = prompts[current_index]
    prompt_id = current_prompt['id']
    
    # Save the current response associated with its prompt ID
    session['responses'][prompt_id] = data
    
    # Prepare for the next prompt
    next_index = current_index + 1
    session['current_prompt_index'] = next_index
    
    if next_index >= len(prompts):
        # All prompts completed
        # Save all responses to a single JSON file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f'typing_metrics_{timestamp}.json'
        filepath = os.path.join(DATA_DIR, filename)
        
        response_data = {
            'session_start_time': data.get('overall_metrics', {}).get('start_time', ''),
            'session_end_time': data.get('overall_metrics', {}).get('end_time', ''),
            'prompt_responses': session.get('responses', {}),
            'total_prompts': len(prompts)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(response_data, f, indent=4)
            # Clear session
            session.pop('prompts', None)
            session.pop('current_prompt_index', None)
            session.pop('responses', None)
            return jsonify({'status': 'completed', 'filename': filename}), 200
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        # Send the next prompt after a 2-second delay on the frontend
        next_prompt = prompts[next_index]
        return jsonify({
            'status': 'success',
            'next_prompt': next_prompt['text'],
            'next_prompt_id': next_prompt['id'],
            'current_prompt_number': next_index + 1
        }), 200

if __name__ == '__main__':
    app.run(debug=True)
