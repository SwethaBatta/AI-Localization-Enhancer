# AI Translation Enhancer (Gradio)

This version uses **Gradio** instead of Streamlit to build an English-to-German translation UI using Hugging Face's MarianMT.

## Setup Instructions

1. Unzip or clone this repo.
2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   python app/main.py
   ```

Then open your browser at the provided `http://localhost:7860`.

## Requirements

- Hugging Face Transformers
- Torch
- Gradio
- SentencePiece
