from flask import Flask, render_template, request, jsonify, session
import json
import os
from datetime import datetime
from flask import redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key in production

# Directory to store typing metrics
DATA_DIR = 'typing_data'
os.makedirs(DATA_DIR, exist_ok=True)

# Path to the prompts file
PROMPTS_FILE = 'prompts.txt'

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
