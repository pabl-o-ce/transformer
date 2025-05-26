import spaces
import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Configuration
MODEL_ID = "somosnlp-hackathon-2025/mistral-7B-ec-es-recetas"
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 512
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# Global variables
model = None
tokenizer = None

css = """
.bubble-wrap {
    padding-top: calc(var(--spacing-xl) * 3) !important;
}
.message-row {
    justify-content: space-evenly !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: calc(var(--spacing-xl)) 0 !important;
    padding: 0 calc(var(--spacing-xl) * 3) !important;
}
.flex-wrap.user {
    border-bottom-right-radius: var(--radius-lg) !important;
}
.flex-wrap.bot {
    border-bottom-left-radius: var(--radius-lg) !important;
}
.message.user{
    padding: 10px;
}
.message.bot{
    text-align: right;
    width: 100%;
    padding: 10px;
    border-radius: 10px;
}
.message-bubble-border {
    border-radius: 6px !important;
}
.message-buttons {
    justify-content: flex-end !important;
}
.message-buttons-left {
    align-self: end !important;
}
.message-buttons-bot, .message-buttons-user {
    right: 10px !important;
    left: auto !important;
    bottom: 2px !important;
}
.dark.message-bubble-border {
    border-color: #343140 !important;
}
.dark.user {
    background: #1e1c26 !important;
}
.dark.assistant.dark, .dark.pending.dark {
    background: #16141c !important;
}
"""

def load_model():
    """Load model and tokenizer"""
    global model, tokenizer
    
    if torch.cuda.is_available():
        print(f"Loading model: {MODEL_ID}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("❌ CUDA not available")
        return False

# Load model on startup
model_loaded = load_model()

@spaces.GPU
def generate(
    message: str,
    chat_history: list[tuple],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """Generate response with streaming"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        yield "❌ Error: Modelo no disponible. Por favor, reinicia la aplicación."
        return
    
    # Convert chat_history format from tuples to messages
    conversation = []
    for user_msg, assistant_msg in chat_history:
        conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    conversation.append({"role": "user", "content": message})
    
    try:
        # Apply chat template
        input_ids = tokenizer.apply_chat_template(
            conversation, 
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        # Check input length
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Conversación recortada a {MAX_INPUT_TOKEN_LENGTH} tokens.")
        
        input_ids = input_ids.to(model.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            tokenizer, 
            timeout=30.0, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Generation parameters
        generate_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Start generation in separate thread
        generation_thread = Thread(target=model.generate, kwargs=generate_kwargs)
        generation_thread.start()
        
        # Stream response
        outputs = []
        try:
            for new_text in streamer:
                outputs.append(new_text)
                yield "".join(outputs)
        except Exception as e:
            yield f"❌ Error durante la generación: {str(e)}"
        finally:
            generation_thread.join(timeout=1)
            
    except Exception as e:
        yield f"❌ Error: {str(e)}"

PLACEHOLDER = """
<div class="message-bubble-border" style="display:flex; max-width: 600px; border-radius: 6px; border-width: 1px; border-color: #e5e7eb; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); backdrop-filter: blur(10px);">
    <figure style="margin: 0;max-width: 200px;min-height: 300px;">
        <img src="https://huggingface.co/spaces/pabloce/llama-cpp-agent/resolve/main/llama.jpg" alt="Logo" style="width: 100%; height: 100%; border-radius: 8px;">
    </figure>
    <div style="padding: .5rem 1.5rem;display: flex;flex-direction: column;justify-content: space-evenly;">
        <h2 style="text-align: left; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">Gastronomia hispana</h2>
        <p style="text-align: left; font-size: 16px; line-height: 1.5; margin-bottom: 15px;">HackathonSomosNLP 2025: Impulsando la creación de modelos de lenguaje alineados con la cultura de los países de LATAM y la Península Ibérica.</p>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; flex-flow: column; justify-content: space-between;">
                <span style="display: inline-flex; align-items: center; border-radius: 0.375rem; background-color: rgba(229, 70, 77, 0.1); padding: 0.1rem 0.75rem; font-size: 0.75rem; font-weight: 500; color: #f88181; margin-bottom: 2.5px;">
                    somosnlp-hackathon-2025/mistral-7B-ec-es-recetas
                </span>
            </div>
            <div style="display: flex; justify-content: flex-end; align-items: center;">
                <a href="https://discord.gg/fgr5RycPFP" target="_blank" rel="noreferrer" style="padding: .5rem;">
                    <svg width="24" height="24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 5 30.67 23.25">
                        <title>Discord</title>
                        <path d="M26.0015 6.9529C24.0021 6.03845 21.8787 5.37198 19.6623 5C19.3833 5.48048 19.0733 6.13144 18.8563 6.64292C16.4989 6.30193 14.1585 6.30193 11.8336 6.64292C11.6166 6.13144 11.2911 5.48048 11.0276 5C8.79575 5.37198 6.67235 6.03845 4.6869 6.9529C0.672601 12.8736 -0.41235 18.6548 0.130124 24.3585C2.79599 26.2959 5.36889 27.4739 7.89682 28.2489C8.51679 27.4119 9.07477 26.5129 9.55525 25.5675C8.64079 25.2265 7.77283 24.808 6.93587 24.312C7.15286 24.1571 7.36986 23.9866 7.57135 23.8161C12.6241 26.1255 18.0969 26.1255 23.0876 23.8161C23.3046 23.9866 23.5061 24.1571 23.7231 24.312C22.8861 24.808 22.0182 25.2265 21.1037 25.5675C21.5842 26.5129 22.1422 27.4119 22.7621 28.2489C25.2885 27.4739 27.8769 26.2959 30.5288 24.3585C31.1952 17.7559 29.4733 12.0212 26.0015 6.9529ZM10.2527 20.8402C8.73376 20.8402 7.49382 19.4608 7.49382 17.7714C7.49382 16.082 8.70276 14.7025 10.2527 14.7025C11.7871 14.7025 13.0425 16.082 13.0115 17.7714C13.0115 19.4608 11.7871 20.8402 10.2527 20.8402ZM20.4373 20.8402C18.9183 20.8402 17.6768 19.4608 17.6768 17.7714C17.6768 16.082 18.8873 14.7025 20.4373 14.7025C21.9717 14.7025 23.2271 16.082 23.1961 17.7714C23.1961 19.4608 21.9872 20.8402 20.4373 20.8402Z"></path>
                    </svg>
                </a>
                <a href="https://github.com/Maximilian-Winter/llama-cpp-agent" target="_blank" rel="noreferrer" style="padding: .5rem;">
                    <svg width="24" height="24" fill="currentColor" viewBox="3 3 18 18">
                        <title>GitHub</title>
                        <path d="M12 3C7.0275 3 3 7.12937 3 12.2276C3 16.3109 5.57625 19.7597 9.15374 20.9824C9.60374 21.0631 9.77249 20.7863 9.77249 20.5441C9.77249 20.3249 9.76125 19.5982 9.76125 18.8254C7.5 19.2522 6.915 18.2602 6.735 17.7412C6.63375 17.4759 6.19499 16.6569 5.8125 16.4378C5.4975 16.2647 5.0475 15.838 5.80124 15.8264C6.51 15.8149 7.01625 16.4954 7.18499 16.7723C7.99499 18.1679 9.28875 17.7758 9.80625 17.5335C9.885 16.9337 10.1212 16.53 10.38 16.2993C8.3775 16.0687 6.285 15.2728 6.285 11.7432C6.285 10.7397 6.63375 9.9092 7.20749 9.26326C7.1175 9.03257 6.8025 8.08674 7.2975 6.81794C7.2975 6.81794 8.05125 6.57571 9.77249 7.76377C10.4925 7.55615 11.2575 7.45234 12.0225 7.45234C12.7875 7.45234 13.5525 7.55615 14.2725 7.76377C15.9937 6.56418 16.7475 6.81794 16.7475 6.81794C17.2424 8.08674 16.9275 9.03257 16.8375 9.26326C17.4113 9.9092 17.76 10.7281 17.76 11.7432C17.76 15.2843 15.6563 16.0687 13.6537 16.2993C13.98 16.5877 14.2613 17.1414 14.2613 18.0065C14.2613 19.2407 14.25 20.2326 14.25 20.5441C14.25 20.7863 14.4188 21.0746 14.8688 20.9824C16.6554 20.364 18.2079 19.1866 19.3078 17.6162C20.4077 16.0457 20.9995 14.1611 21 12.2276C21 7.12937 16.9725 3 12 3Z"></path>
                    </svg>
                </a>
            </div>
        </div>
    </div>
</div>
"""

# Create ChatInterface directly (no Blocks wrapper)
demo = gr.ChatInterface(
    fn=generate,
    description="Recetas en español",
    chatbot=gr.Chatbot(
        height=500,
        avatar_images=(None, "🍽️"),
        scale=1, 
        placeholder=PLACEHOLDER,
        likeable=False,
        show_copy_button=True
    ),
    textbox=gr.Textbox(
        placeholder="Escribe tu pregunta sobre recetas, ingredientes o técnicas culinarias...",
        scale=7
    ),
    additional_inputs=[
        gr.Slider(
            label="Longitud máxima de respuesta",
            minimum=100,
            maximum=MAX_MAX_NEW_TOKENS,
            step=50,
            value=DEFAULT_MAX_NEW_TOKENS,
            info="Controla qué tan larga puede ser la respuesta"
        ),
        gr.Slider(
            label="Creatividad (Temperature)",
            minimum=0.1,
            maximum=2.0,
            step=0.1,
            value=0.7,
            info="Más alto = respuestas más creativas, más bajo = más conservadoras"
        ),
        gr.Slider(
            label="Diversidad (Top-p)",
            minimum=0.1,
            maximum=1.0,
            step=0.05,
            value=0.9,
            info="Controla la diversidad en la selección de palabras"
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=100,
            step=1,
            value=50,
            info="Número de opciones de palabras a considerar"
        ),
        gr.Slider(
            label="Penalización por repetición",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
            info="Evita que el modelo repita frases"
        ),
    ],
    examples=[
        ["¿Cuáles son los ingredientes principales del locro ecuatoriano?"],
        ["Dame una receta completa de sancocho de gallina criolla"],
        ["¿Cómo hacer empanadas de verde ecuatorianas?"],
    ],
    cache_examples=False,
    retry_btn="Reintentar",
    undo_btn="Deshacer",
    clear_btn="Limpiar",
    submit_btn="Enviar",
    stop_btn="Detener",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="violet", neutral_hue="gray",font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"]).set(
        body_background_fill_dark="#16141c",
        block_background_fill_dark="#16141c",
        block_border_width="1px",
        block_title_background_fill_dark="#1e1c26",
        input_background_fill_dark="#292733",
        button_secondary_background_fill_dark="#24212b",
        border_color_accent_dark="#343140",
        border_color_primary_dark="#343140",
        background_fill_secondary_dark="#16141c",
        color_accent_soft_dark="transparent",
        code_background_fill_dark="#292733",
    ),
    css=css
)

if __name__ == "__main__":
    if model_loaded:
        print("🚀 Launching Gradio app...")
        demo.launch(
            share=False,
            show_error=True
        )
    else:
        print("❌ Failed to load model. Cannot start the app.")