import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore")

from threading import Thread
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
model = model.to(device)

chat = [
    { "role": "user", "content": "Describe the sights and sounds of a bustling farmers' market. Use around 50 words."},
]
question = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

question = tokenizer(question, return_tensors="pt").to(device)

streamer = TextStreamer(tokenizer, skip_prompt=True)

_ = model.generate(
    **question, 
    streamer=streamer,
    pad_token_id=tokenizer.eos_token_id, 
    max_length=128, 
    temperature=0.5,
    do_sample=True,
    top_p=0.8,
    repetition_penalty=1.25
)

print(streamer.get_output())



# # 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