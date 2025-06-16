import spaces
import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Configuration
MODEL_ID = "somosnlp-hackathon-2025/mistral-7b-gastronomia-hispana-qlora-v1"
MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 2048
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# Global variables
model = None
tokenizer = None

css = """
.bubble-wrap {
    padding-top: calc(var(--spacing-xl) * 3) !important;
    border-color: #1f2b21 !important;
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
    border-color: #1f2b21 !important;
}
.dark.user {
    background: #202721 !important;
}
.dark.assistant.dark, .dark.pending.dark {
    background: #202721 !important;
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
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )

            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print("CUDA not available")
        return False


# Load model on startup
model_loaded = load_model()


@spaces.GPU
def generate(
    message: str,
    history: list[tuple],
    system_message: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.2,
):
    """Generate response with streaming"""
    global model, tokenizer

    if model is None or tokenizer is None:
        yield "Error: Modelo no disponible. Por favor, reinicia la aplicación."
        return

    # Convert chat_history format from tuples to messages
    conversation = []
    # Add system prompt if provided
    # if system_message:
    #     conversation.append({"role": "system", "content": system_message})

    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:
            conversation.append({"role": "assistant", "content": assistant_msg})

    # Add current message
    conversation.append({"role": "user", "content": message})

    try:
        # Apply chat template
        input_ids = tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True
        )

        # Check input length
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Conversación recortada a {MAX_INPUT_TOKEN_LENGTH} tokens.")

        input_ids = input_ids.to(model.device)

        attention_mask = torch.ones_like(input_ids, device=model.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True
        )

        # Generation parameters
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": False,
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
            yield f"Error durante la generación: {str(e)}"
        finally:
            generation_thread.join(timeout=1)

    except Exception as e:
        yield f"Error: {str(e)}"


PLACEHOLDER = f"""
<div style="display: flex; flex-wrap: wrap; max-width: min(600px, 95vw); width: 100%; border-radius: clamp(6px, 1vw, 6px); border-color: var(--border-color-primary); border-width: 1px; box-shadow: 0 4px 6px rgba(36, 210, 0, 0.16); backdrop-filter: blur(10px); overflow: hidden; transition: all 0.3s ease;">
    <div style="flex: 1 1 clamp(200px, 40%, 250px); min-height: clamp(200px, 30vh, 300px); position: relative; overflow: hidden;">
        <img src="https://raw.githubusercontent.com/pabl-o-ce/transformer/main/logo.gif" alt="Gastronomia Hispana Logo" style="width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s ease;">
    </div>
    <div style="flex: 1 1 clamp(280px, 60%, 400px); padding: clamp(1rem, 3vw, 2rem); display: flex; flex-direction: column; justify-content: space-between;">
        <div style="flex-grow: 1;">
            <h2 style="font-size: clamp(1.125rem, 4vw, 1.75rem); font-weight: 700; margin: 0 0 clamp(0.5rem, 2vw, 1rem) 0; line-height: 1.2;">Gastronomia hispana</h2>
            <p style="font-size: clamp(0.875rem, 3vw, 1rem); line-height: 1.6; text-align: left; margin: 0 0 clamp(1rem, 3vw, 1.5rem) 0;">HackathonSomosNLP 2025: Impulsando la creación de modelos de lenguaje alineados con la cultura de los países de LATAM y la Península Ibérica.</p>
        </div>
        <div style="display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: flex-end; gap: clamp(0.5rem, 2vw, 1rem);">
            <div style="flex: 1 1 200px; min-width: 0;">
                <span style="display: inline-block; border-radius: clamp(0.375rem, 1vw, 0.375rem); background-color: rgba(75, 229, 70, 0.1); padding: clamp(0.1rem, 1vw, 0.1rem) clamp(0.75rem, 2vw, 0.75rem); font-size: clamp(0.625rem, 2vw, 0.75rem); font-weight: 500; color: #9bf881; word-break: break-word; max-width: 100%;">
                    {MODEL_ID}
                </span>
            </div>
            <div style="display: flex; gap: clamp(0.25rem, 1vw, 0.5rem); flex-shrink: 0;">
                <a href="https://discord.com/invite/my8w7JUxZR" target="_blank" rel="noreferrer" style="display: flex; align-items: center; justify-content: center; width: clamp(32px, 8vw, 44px); height: clamp(32px, 8vw, 44px); border-radius: 20%; background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color); text-decoration: none; transition: all 0.3s ease;">
                    <svg style="width: clamp(16px, 4vw, 22px); height: clamp(16px, 4vw, 22px);" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 5 30.67 23.25">
                        <path d="M26.0015 6.9529C24.0021 6.03845 21.8787 5.37198 19.6623 5C19.3833 5.48048 19.0733 6.13144 18.8563 6.64292C16.4989 6.30193 14.1585 6.30193 11.8336 6.64292C11.6166 6.13144 11.2911 5.48048 11.0276 5C8.79575 5.37198 6.67235 6.03845 4.6869 6.9529C0.672601 12.8736 -0.41235 18.6548 0.130124 24.3585C2.79599 26.2959 5.36889 27.4739 7.89682 28.2489C8.51679 27.4119 9.07477 26.5129 9.55525 25.5675C8.64079 25.2265 7.77283 24.808 6.93587 24.312C7.15286 24.1571 7.36986 23.9866 7.57135 23.8161C12.6241 26.1255 18.0969 26.1255 23.0876 23.8161C23.3046 23.9866 23.5061 24.1571 23.7231 24.312C22.8861 24.808 22.0182 25.2265 21.1037 25.5675C21.5842 26.5129 22.1422 27.4119 22.7621 28.2489C25.2885 27.4739 27.8769 26.2959 30.5288 24.3585C31.1952 17.7559 29.4733 12.0212 26.0015 6.9529ZM10.2527 20.8402C8.73376 20.8402 7.49382 19.4608 7.49382 17.7714C7.49382 16.082 8.70276 14.7025 10.2527 14.7025C11.7871 14.7025 13.0425 16.082 13.0115 17.7714C13.0115 19.4608 11.7871 20.8402 10.2527 20.8402ZM20.4373 20.8402C18.9183 20.8402 17.6768 19.4608 17.6768 17.7714C17.6768 16.082 18.8873 14.7025 20.4373 14.7025C21.9717 14.7025 23.2271 16.082 23.1961 17.7714C23.1961 19.4608 21.9872 20.8402 20.4373 20.8402Z"></path>
                    </svg>
                </a>
                <a href="https://somosnlp.org/" target="_blank" rel="noreferrer" style="display: flex; align-items: center; justify-content: center; width: clamp(32px, 8vw, 44px); height: clamp(32px, 8vw, 44px); border-radius: 20%; background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color); text-decoration: none; transition: all 0.3s ease;">
                    <svg style="width: clamp(16px, 4vw, 22px); height: clamp(16px, 4vw, 22px);" stroke="currentColor" fill="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m6.115 5.19.319 1.913A6 6 0 0 0 8.11 10.36L9.75 12l-.387.775c-.217.433-.132.956.21 1.298l1.348 1.348c.21.21.329.497.329.795v1.089c0 .426.24.815.622 1.006l.153.076c.433.217.956.132 1.298-.21l.723-.723a8.7 8.7 0 0 0 2.288-4.042 1.087 1.087 0 0 0-.358-1.099l-1.33-1.108c-.251-.21-.582-.299-.905-.245l-1.17.195a1.125 1.125 0 0 1-.98-.314l-.295-.295a1.125 1.125 0 0 1 0-1.591l.13-.132a1.125 1.125 0 0 1 1.3-.21l.603.302a.809.809 0 0 0 1.086-1.086L14.25 7.5l1.256-.837a4.5 4.5 0 0 0 1.528-1.732l.146-.292M6.115 5.19A9 9 0 1 0 17.18 4.64M6.115 5.19A8.965 8.965 0 0 1 12 3c1.929 0 3.716.607 5.18 1.64" />
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
        scale=1,
        placeholder=PLACEHOLDER,
        likeable=False,
        show_copy_button=True,
    ),
    textbox=gr.Textbox(
        placeholder="Escribe tu pregunta sobre recetas, ingredientes o técnicas culinarias...",
        scale=7,
    ),
    additional_inputs=[
        gr.Textbox(
            value="Eres un maestro culinario especializado en técnicas de cocción internacionales, con expertise en tiempos, temperaturas y métodos tradicionales de diversas culturas gastronómicas.",
            label="System message",
        ),
        gr.Slider(
            label="Longitud máxima de respuesta",
            minimum=100,
            maximum=MAX_MAX_NEW_TOKENS,
            step=50,
            value=DEFAULT_MAX_NEW_TOKENS,
            info="Controla qué tan larga puede ser la respuesta",
        ),
        gr.Slider(
            label="Creatividad (Temperature)",
            minimum=0.1,
            maximum=2.0,
            step=0.1,
            value=0.3,
            info="Más alto = respuestas más creativas, más bajo = más conservadoras",
        ),
        gr.Slider(
            label="Diversidad (Top-p)",
            minimum=0.1,
            maximum=1.0,
            step=0.05,
            value=0.8,
            info="Controla la diversidad en la selección de palabras",
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=100,
            step=1,
            value=20,
            info="Número de opciones de palabras a considerar",
        ),
        gr.Slider(
            label="Penalización por repetición",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
            info="Evita que el modelo repita frases",
        ),
    ],
    examples=[
        ["¿Podrías explicarme paso a paso cómo preparar encebollado ecuatoriano?"],
        [
            "¿Cuál es la importancia cultural de la colada morada en Ecuador y cuándo se prepara tradicionalmente?"
        ],
        [
            "¿Cuál es la técnica correcta para freír pescado para un encocado sin que se desbarate?"
        ],
    ],
    cache_examples=False,
    retry_btn="Reintentar",
    undo_btn="Deshacer",
    clear_btn="Limpiar",
    submit_btn="Enviar",
    stop_btn="Detener",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill_dark="#171717",
        block_background_fill_dark="#171717",
        block_border_width="1px",
        block_title_background_fill_dark="#1d1d1d",
        input_background_fill_dark="#1e1e1e",
        button_secondary_background_fill_dark="#1d1d1d",
        border_color_accent_dark="#1f2b21",
        border_color_primary_dark="#1f2b21",
        background_fill_secondary_dark="#171717",
        color_accent_soft_dark="transparent",
        code_background_fill_dark="#1e1e1e",
    ),
    css=css,
)

if __name__ == "__main__":
    if model_loaded:
        print("Launching Gradio app...")
        demo.launch(share=False, show_error=True)
    else:
        print("Failed to load model. Cannot start the app.")
