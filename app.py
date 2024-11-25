from flask import Flask, render_template, request
from diffusers import StableDiffusionPipeline
import torch
from groq import Groq
import os
client = Groq(api_key = "env/key")
MODEL = 'Llama-3.1-8b-instant'

app = Flask(__name__)
model_path = "path_to_your_saved_model"
pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    if request.method == "POST":
        user_prompt = request.form.get("textbox")
        messages=[
            {
                "role": "system",
                "content": "Your primary role is to assist Stable diffusion by providing concise and accurate promts only related to helping create an image.Answer should be in a form which is in detail and useful for image generation. Refuse to answer questions on any other topic other than hospital work. This includes, but is not limited to, diagnosing medical conditions based on symptoms, suggesting potential treatment plans, and providing general medical information. Please note that while you strive to provide reliable information, your responses should not replace professional medical advice."
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tool_choice="auto",  
            max_tokens=4096
        )
        response_message = response.choices[0].message.content
        image = pipeline(response_message).images[0]
        image_filename = "generated_image.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)
    return render_template("index.html", image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
