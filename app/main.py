import gradio as gr
from transformers import MarianMTModel, MarianTokenizer
import torch
from bleurt import score as bleurt_score
import pyttsx3

# Define language mappings
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta"
}

# Reverse map for dropdown
language_names = list(LANGUAGES.keys())

# Load BLEURT scorer
# bleurt_model_path = "bleurt/bleurt-base-128"  # Adjust if needed
# bleurt_scorer = bleurt_score.BleurtScorer(bleurt_model_path)

# After download and extraction
bleurt_model_path = "bleurt/BLEURT-20"
bleurt_scorer = bleurt_score.BleurtScorer(bleurt_model_path)

# TTS engine
tts_engine = pyttsx3.init()

def get_model_name(src, tgt):
    return f'Helsinki-NLP/opus-mt-{src}-{tgt}'

def get_valid_targets(src_lang):
    return [lang for lang in language_names if lang != src_lang]

def translate(text, src_lang, tgt_lang):
    if not text or not src_lang or not tgt_lang:
        return "", "Missing input."

    src = LANGUAGES[src_lang]
    tgt = LANGUAGES[tgt_lang]

    model_name = get_model_name(src, tgt)
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except:
        return "", f"Translation model for {src_lang} → {tgt_lang} not available."

    # Translate
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # Score with BLEURT
    score = bleurt_scorer.score(references=[text], candidates=[translated_text])
    score = round(score[0], 3)

    return translated_text, score

def update_target_choices(src_lang):
    return gr.Dropdown.update(choices=get_valid_targets(src_lang), value=None)

def play_tts(text):
    tts_engine.say(text)
    tts_engine.runAndWait()
    return "🔊 Playing..."

def score_legend():
    return """🔍 **BLEURT Score Legend**
- **0.80 – 1.00**: Excellent translation
- **0.60 – 0.79**: Good, minor issues
- **0.40 – 0.59**: Fair, some errors
- **Below 0.40**: Poor quality
"""

# UI
with gr.Blocks() as demo:
    gr.Markdown("# 🌐 AI Translation Enhancer with BLEURT")
    gr.Markdown("Translate text between languages, listen to output, and evaluate translation quality.")

    with gr.Row():
        source_lang = gr.Dropdown(label="Source Language", choices=language_names, value="English")
        target_lang = gr.Dropdown(label="Target Language", choices=get_valid_targets("English"))

    source_lang.change(fn=update_target_choices, inputs=source_lang, outputs=target_lang)

    input_text = gr.Textbox(label="Enter text to translate", lines=3)
    translate_btn = gr.Button("Translate")
    translated_text = gr.Textbox(label="Translated Output", lines=3)
    tts_btn = gr.Button("🔊 Listen to Translation")
    score_label = gr.Label(label="BLEURT Quality Score")

    with gr.Accordion("BLEURT Score Legend", open=False):
        gr.Markdown(score_legend())

    translate_btn.click(fn=translate,
                        inputs=[input_text, source_lang, target_lang],
                        outputs=[translated_text, score_label])

    tts_btn.click(fn=play_tts, inputs=translated_text, outputs=None)

if __name__ == "__main__":
    demo.launch()
