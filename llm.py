import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore")

from threading import Thread
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
# model = model.to(device)

# chat = [
#     { "role": "user", "content": "Describe the sights and sounds of a bustling farmers' market."},
# ]
# question = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# question = tokenizer(question, return_tensors="pt").to(device)

# streamer = TextStreamer(tokenizer, skip_prompt=True)

# _ = model.generate(
#     **question, 
#     streamer=streamer,
#     pad_token_id=tokenizer.eos_token_id, 
#     max_length=128, 
#     temperature=0.5,
#     do_sample=True,
#     top_p=0.8,
#     repetition_penalty=1.25
# )

# print(streamer.get_output())



# # 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


from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

chat = [
    { "role": "user", "content": "Describe the sights and sounds of a bustling farmers' market." },
]
inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(inputs, return_tensors="pt")

streamer = TextIteratorStreamer(tokenizer)

# Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=128)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print("-----")
start_add_to_response = False
original_text = ""
generated_text = ""
for new_text in streamer:
    original_text += new_text
    new_text = new_text.replace("\n", " ").replace("<eos>", "").replace("**", " ").replace("*", " ")
    if "<start_of_turn>model" in new_text:
        start_add_to_response = True
        continue

    if not start_add_to_response:
        continue

    generated_text += new_text
    print(new_text, end="", flush=True)

thread.join()

print("-----")
print("Original text:\n", original_text)
print("Generated text:\n", generated_text)