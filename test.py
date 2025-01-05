# from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextIteratorStreamer
# import threading

# # Load pre-trained tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # Define the prompt
# prompt = "Once upon a time, in a distant galaxy,"

# # Tokenize the input prompt
# encoded_input = tokenizer(prompt, return_tensors="pt")

# # Create a TextIteratorStreamer for real-time streaming
# streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# # Start a new thread to run the generation
# def generate_text():
#     model.generate(
#         input_ids=encoded_input["input_ids"],  # Input prompt
#         max_length=50,                        # Max length of generated text
#         streamer=streamer,                    # Use the streamer for real-time output
#         temperature=0.7,                      # Controls randomness
#         top_k=50,                             # Top-k sampling
#         top_p=0.95,                           # Nucleus sampling
#         do_sample=True                        # Enable sampling for diversity
#     )

# # Run the generation in a separate thread
# generation_thread = threading.Thread(target=generate_text)
# generation_thread.start()

# # Stream and print the tokens in real time
# print("Generated text:")
# for token in streamer:
#     print(token, end="", flush=True)  # Print the token immediately without newline

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextIteratorStreamer
import threading

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large")

# Set the pad token to the EOS token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Define the prompt
prompt = "Describe the sights and sounds of a bustling farmers' market. Use around 50 words."

# Tokenize the input prompt
encoded_input = tokenizer(prompt, return_tensors="pt", padding=True)

# Create a TextIteratorStreamer for real-time streaming
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Start a new thread to run the generation
def generate_text():
    model.generate(
        input_ids=encoded_input["input_ids"],   # Input prompt
        attention_mask=encoded_input["attention_mask"],  # Include attention mask
        max_length=512,                         # Max length of generated text
        streamer=streamer,                     # Use the streamer for real-time output
        temperature=0.7,                       # Controls randomness
        top_k=50,                              # Top-k sampling
        top_p=0.95,                            # Nucleus sampling
        do_sample=True                         # Enable sampling for diversity
    )

# Run the generation in a separate thread
generation_thread = threading.Thread(target=generate_text)
generation_thread.start()

# Stream and print the tokens in real time
print("Generated text:")
for token in streamer:
    print(token, end="", flush=True)  # Print the token immediately without newline

