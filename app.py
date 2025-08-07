import gradio as gr
from transformers import pipeline, BitsAndBytesConfig
from peft import PeftModel  # Optional if you load fine-tuned adapters
import torch
from PIL import Image
from gtts import gTTS
import whisper
import tempfile
import os

# --------- Load Model ---------
model_id = "google/medgemma-4b-it"
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": "auto"
}

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model_kwargs["quantization_config"] = bnb_config

pipe = pipeline("image-text-to-text", model=model_id, model_kwargs=model_kwargs)
pipe.model.generation_config.do_sample = False

# --------- Whisper Model ---------
whisper_model = whisper.load_model("base")

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def generate_tts(text):
    tts = gTTS(text)
    tts_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tts_file.name)
    return tts_file.name

# --------- Inference Function ---------
def process_inputs(audio_path, image):
    prompt = transcribe_audio(audio_path)

    role_instruction = "You are an expert dermatologist. Provide a clinical report."
    messages = [
        {"role": "system", "content": [{"type": "text", "text": role_instruction}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}
        ]}
    ]

    output = pipe(text=messages, max_new_tokens=500)
    response = output[0]["generated_text"][-1]["content"]
    response_audio = generate_tts(response)

    return prompt, response, response_audio

# --------- Gradio UI ---------
def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧠 Skin Diagnosis Assistant (Voice + Image + AI Response)")

        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="🎤 Upload Your Voice (.wav or .mp3)")
            image_input = gr.Image(type="filepath", label="📷 Upload Skin Image")

        transcribed = gr.Textbox(label="📝 Transcribed Prompt")
        response = gr.Textbox(label="📋 AI Diagnosis")
        voice_output = gr.Audio(label="🔊 AI Voice Response")

        submit = gr.Button("🧪 Diagnose")

        submit.click(
            fn=process_inputs,
            inputs=[audio_input, image_input],
            outputs=[transcribed, response, voice_output]
        )

    demo.launch()

if __name__ == "__main__":
    launch_app()
