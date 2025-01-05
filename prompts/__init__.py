import os


PROMPTS_FILE = 'prompts/prompts.txt'


def load_prompts():
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