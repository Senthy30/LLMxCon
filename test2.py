from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained tokenizer and model with a language modeling head
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

# Define your prompt
prompt = "Once upon a time, in a distant galaxy,"

# Tokenize the input prompt
encoded_input = tokenizer(prompt, return_tensors='pt')

# Generate text
output = model.generate(
    input_ids=encoded_input['input_ids'],  # Input prompt
    max_length=50,                        # Max length of the generated text
    num_return_sequences=1,               # Number of sequences to generate
    temperature=0.7,                      # Controls randomness in predictions
    top_k=50,                             # Top-k sampling
    top_p=0.95,                           # Nucleus sampling
    do_sample=True                        # Enable sampling for diversity
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
