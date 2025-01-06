import openai
from openai import AzureOpenAI  
import time

prompts = []
with open('prompts.txt', 'r') as file:
    i = 0
    for line in file:
        i+=1
        prompts.append(line)

endpoint =  "SECRET"
deployment = "gpt-4"  
subscription_key = "SECRET"

client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",  
)

def generate_pipeline(prompt):
    start_time = time.time()
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"Respond to the following prompt: generate a phrase of maximum 25 words based on this description: {prompt}"
                }
            ]
        }
    ] 
        
    messages = chat_prompt  
        
    completion = client.chat.completions.create(  
        model=deployment,  
        messages=messages,  
        max_tokens=800,  
        temperature=0.7,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,  
        stop=None,  
        stream=False
    )

    response_time = time.time() - start_time  

    return {"response_time": response_time, "text": completion.choices[0].message.content, "length": len(completion.choices[0].message.content)}

import json 

responses = {}

for prompt in prompts:
    result = generate_pipeline(prompt)
    responses[prompt.strip()] = result
    print(f"Processed prompt: {prompt.strip()} with response time: {result['response_time']} seconds and response text length: {len(result['text'])}")


with open('responses.json', 'w') as json_file:
    json.dump(responses, json_file, indent=4)