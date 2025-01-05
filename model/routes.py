import os
import torch

from threading import Thread
from prompts import load_prompts
from flask import Blueprint, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


model_routes = Blueprint('model_routes', __name__)

next_model_response_id = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_responses = []
prompts = load_prompts()


@model_routes.route('/nextModelResponseId', methods=['GET'])
def get_next_model_response_id():
    global next_model_response_id
    next_model_response_id += 1
    return jsonify({'response_id': next_model_response_id})


@model_routes.route('/getModelResponses', methods=['GET'])
def get_model_responses():
    return jsonify(model_responses)


@model_routes.route('/getModelResponse/<int:response_id>', methods=['GET'])
def get_model_response(response_id):
    for model_response in model_responses:
        if model_response['response_id'] == response_id:
            return jsonify(model_response)
    return jsonify({'error': 'Model response not found.'}), 404


@model_routes.route('/getPrompt', methods=['GET'])
def get_prompt():
    if not prompts:
        return jsonify({'status': 'error', 'message': 'No prompts available. Please add prompts to prompts.txt.'}), 404
    return jsonify({'status': 'success', 'prompts': prompts})


@model_routes.route('/getPrompt/<prompt_id>', methods=['GET'])
def get_prompt_by_id(prompt_id):
    if not prompts:
        return jsonify({'status': 'error', 'message': 'No prompts available. Please add prompts to prompts.txt.'}), 404
    for prompt in prompts:
        if prompt['id'] == prompt_id:
            return jsonify({'status': 'success', 'prompt': prompt})
    return jsonify({'status': 'error', 'message': 'Prompt not found.'}), 404


@model_routes.route('/generateModelResponse', methods=['POST'])
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

    question += " Use around 50 words."
    chat = [
        { "role": "user", "content": question},
    ]

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

    inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt")

    streamer = TextIteratorStreamer(tokenizer)

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=256)
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

