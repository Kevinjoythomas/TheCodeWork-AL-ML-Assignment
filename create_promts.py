from dotenv import load_dotenv
import os
from transformers import pipeline
from groq import Groq
client = Groq(api_key = "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")

MODEL = 'Llama-3.1-8b-instant'


def generate_text_prompt_with_gpt(objects):
    prompt = []
    for i in range(len(objects['category'])):
        element = objects['category'][i] if objects['category'][i] is not None else "N/A"
        color = objects['color'][i] if objects['color'][i] is not None else "N/A"
        radius = objects['radius'][i] if objects['radius'][i] is not None else "N/A"
        text = objects['text'][i] if objects['text'][i] is not None else "N/A"
        bbox = objects['bbox'][i] if objects['bbox'][i] is not None else "N/A"
        
        user_prompt = f"Create a detailed description for a mobile UI element with the following attributes:\n" \
                      f"Element: {element}, Color: {color}, Radius: {radius}, Text: {text}, Position: {bbox}.\n" \
                      f"Provide a natural and context-rich description for this UI element."
       
        prompt.append(user_prompt)
    
    # Combine descriptions of all UI elements into a full prompt
    return " ".join(prompt)
