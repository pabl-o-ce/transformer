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
                trust_remote_code=True
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
    top_p: float = 0.9,
    top_k: int = 50,
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
    if system_prompt.strip():
        conversation.append({"role": "system", "content": system_message})
    
    for user_msg, assistant_msg in history:
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
            yield f"Error durante la generación: {str(e)}"
        finally:
            generation_thread.join(timeout=1)
            
    except Exception as e:
        yield f"Error: {str(e)}"

PLACEHOLDER = """
<div class="message-bubble-border" style="display:flex; max-width: 600px; border-radius: 6px; border-width: 1px; border-color: #e5e7eb; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); backdrop-filter: blur(10px);">
    <figure style="margin: 0;max-width: 200px;min-height: 300px;">
        <img src="https://cdn-lfs-us-1.hf.co/repos/9f/f8/9ff8df73bcfacf4b841aa5ab4168dbd225d08d8f13c51327247b39932bffa957/9baf6a088e4ab9d035650dc561275cf13ec9498bf4beb5da7eb35748fe5bbddc?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27logo.jpg%3B+filename%3D%22logo.jpg%22%3B&response-content-type=image%2Fjpeg&Expires=1748303740&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0ODMwMzc0MH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzlmL2Y4LzlmZjhkZjczYmNmYWNmNGI4NDFhYTVhYjQxNjhkYmQyMjVkMDhkOGYxM2M1MTMyNzI0N2IzOTkzMmJmZmE5NTcvOWJhZjZhMDg4ZTRhYjlkMDM1NjUwZGM1NjEyNzVjZjEzZWM5NDk4YmY0YmViNWRhN2ViMzU3NDhmZTViYmRkYz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=umbDu%7E6ngGR2Tp25XYNEmioQd5CQvhDFUWrfSQouR9IzrzPw0Jd1z97X-s4rGXgYXqFEB5d93NX9CPZhH5zNBMYdq6DzbogpQYA8VjGfmS9wcf4ua9sBv-8uCcQBRXCXLzK9mAzisLpKOBI-glMKximNUodJpju5Wx3XBcM6rDrL7ljtX7uwI3huzIxCun6aQ1wNti01AlJZ0CEmr3dBHTFpRK3GcquyuxcVGWlOOhnSj0%7EB3vdV70hFxXx-maqmbVjUFDBwmAFsKphIgcGZKTHzq5XGif%7ERSWa6lgE1g4kSDjc3WI9OWDJ7CFLn8ROxam%7EjBvi-45PiwOL9QDJShQ__&Key-Pair-Id=K24J24Z295AEI9" alt="Logo" style="width: 100%; height: 100%; border-radius: 8px;">
    </figure>
    <div style="padding: .5rem 1.5rem;display: flex;flex-direction: column;justify-content: space-evenly;">
        <h2 style="text-align: left; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">Gastronomia hispana</h2>
        <p style="text-align: left; font-size: 16px; line-height: 1.5; margin-bottom: 15px;">HackathonSomosNLP 2025: Impulsando la creación de modelos de lenguaje alineados con la cultura de los países de LATAM y la Península Ibérica.</p>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; flex-flow: column; justify-content: space-between;">
                <span style="display: inline-flex; align-items: center; border-radius: 0.375rem; background-color: rgba(75, 229, 70, 0.1); padding: 0.1rem 0.75rem; font-size: 0.75rem; font-weight: 500; color: #9bf881; margin-bottom: 2.5px;">
                    somosnlp-hackathon-2025/mistral-7B-ec-es-recetas
                </span>
            </div>
            <div style="display: flex; justify-content: flex-end; align-items: center;">
                <a href="https://discord.com/invite/my8w7JUxZR" target="_blank" rel="noreferrer" style="padding: .5rem;">
                    <svg width="24" height="24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 5 30.67 23.25">
                        <title>Discord</title>
                        <path d="M26.0015 6.9529C24.0021 6.03845 21.8787 5.37198 19.6623 5C19.3833 5.48048 19.0733 6.13144 18.8563 6.64292C16.4989 6.30193 14.1585 6.30193 11.8336 6.64292C11.6166 6.13144 11.2911 5.48048 11.0276 5C8.79575 5.37198 6.67235 6.03845 4.6869 6.9529C0.672601 12.8736 -0.41235 18.6548 0.130124 24.3585C2.79599 26.2959 5.36889 27.4739 7.89682 28.2489C8.51679 27.4119 9.07477 26.5129 9.55525 25.5675C8.64079 25.2265 7.77283 24.808 6.93587 24.312C7.15286 24.1571 7.36986 23.9866 7.57135 23.8161C12.6241 26.1255 18.0969 26.1255 23.0876 23.8161C23.3046 23.9866 23.5061 24.1571 23.7231 24.312C22.8861 24.808 22.0182 25.2265 21.1037 25.5675C21.5842 26.5129 22.1422 27.4119 22.7621 28.2489C25.2885 27.4739 27.8769 26.2959 30.5288 24.3585C31.1952 17.7559 29.4733 12.0212 26.0015 6.9529ZM10.2527 20.8402C8.73376 20.8402 7.49382 19.4608 7.49382 17.7714C7.49382 16.082 8.70276 14.7025 10.2527 14.7025C11.7871 14.7025 13.0425 16.082 13.0115 17.7714C13.0115 19.4608 11.7871 20.8402 10.2527 20.8402ZM20.4373 20.8402C18.9183 20.8402 17.6768 19.4608 17.6768 17.7714C17.6768 16.082 18.8873 14.7025 20.4373 14.7025C21.9717 14.7025 23.2271 16.082 23.1961 17.7714C23.1961 19.4608 21.9872 20.8402 20.4373 20.8402Z"></path>
                    </svg>
                </a>
                <a href="https://somosnlp.org/" target="_blank" rel="noreferrer" style="padding: .5rem;">
                    <svg width="24" height="24" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                      <title>Web</title>
                      <path stroke-linecap="round" stroke-linejoin="round" d="m6.115 5.19.319 1.913A6 6 0 0 0 8.11 10.36L9.75 12l-.387.775c-.217.433-.132.956.21 1.298l1.348 1.348c.21.21.329.497.329.795v1.089c0 .426.24.815.622 1.006l.153.076c.433.217.956.132 1.298-.21l.723-.723a8.7 8.7 0 0 0 2.288-4.042 1.087 1.087 0 0 0-.358-1.099l-1.33-1.108c-.251-.21-.582-.299-.905-.245l-1.17.195a1.125 1.125 0 0 1-.98-.314l-.295-.295a1.125 1.125 0 0 1 0-1.591l.13-.132a1.125 1.125 0 0 1 1.3-.21l.603.302a.809.809 0 0 0 1.086-1.086L14.25 7.5l1.256-.837a4.5 4.5 0 0 0 1.528-1.732l.146-.292M6.115 5.19A9 9 0 1 0 17.18 4.64M6.115 5.19A8.965 8.965 0 0 1 12 3c1.929 0 3.716.607 5.18 1.64" />
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
        show_copy_button=True
    ),
    textbox=gr.Textbox(
        placeholder="Escribe tu pregunta sobre recetas, ingredientes o técnicas culinarias...",
        scale=7
    ),
    additional_inputs=[
        gr.Textbox(value="# Sistema de Prompt - Modelo Especializado en Gastronomía Ecuatoriana y Colombiana\n\n## Identidad del Asistente\nEres un asistente culinario experto especializado en la gastronomía de recetas hispanas. Has sido entrenado específicamente con el dataset que contiene conversaciones detalladas sobre recetas, técnicas culinarias y patrimonio gastronómico de estos países. Tu conocimiento se basa en interacciones reales sobre cocina casera, platos tradicionales y adaptaciones modernas.\n\n## Capacidades Principales\n- **Recetas Detalladas**: Proporciona instrucciones paso a paso para platos como Bala de Verde con Encocado de Camarón, Colada Morada, Sancocho, Llapingachos, Chupe de Pescado, y cientos de recetas más\n- **Sustituciones Inteligentes**: Ofreces alternativas precisas cuando faltan ingredientes específicos\n- **Adaptaciones Dietéticas**: Modificas recetas para dietas sin gluten, vegetarianas, bajas en carbohidratos, etc.\n- **Técnicas Culinarias**: Explicas métodos específicos de cocción, horneado, y preparación\n- **Planificación de Comidas**: Ayudas con listas de compras, preparación anticipada y escalado de porciones\n- **Resolución de Problemas**: Diagnosticas y solucionas problemas culinarios comunes\n\n## Estilo de Comunicación\n- Responde siempre en español de manera clara y accesible\n- Usa un tono cálido y entusiasta, con expresiones como "¡Claro que sí!" o "¡Perfecto!"\n- Incluye anécdotas culturales cuando sea relevante\n- Proporciona instrucciones paso a paso extremadamente detalladas\n- Ofrece consejos adicionales marcados como "Consejos:" o "Tips:"\n- Menciona variaciones regionales y contexto cultural\n- Termina con expresiones como "¡Buen provecho!" o "¡Que disfrutes cocinando!"\n- Usa emojis apropiados para hacer las respuestas más amigables\n\n## Categorías de Respuesta Especializadas\n**Según el tipo de consulta, adaptas tu respuesta:**\n- **basic_recipe**: Recetas completas paso a paso\n- **substitutions**: Alternativas de ingredientes con explicaciones detalladas\n- **dietary_modifications**: Adaptaciones para dietas especiales\n- **cooking_techniques**: Explicaciones de métodos de cocción\n- **troubleshooting**: Solución de problemas culinarios\n- **ingredients**: Listas de compras y preparación de ingredientes\n- **time_and_planning**: Planificación de comidas y preparación anticipada\n- **cultural_context**: Historia y significado cultural de los platos\n\n## Recetas Específicas de tu Entrenamiento\n**Incluyes conocimiento detallado sobre:**\n- **Platos Principales**: Sancocho de Res, Chupe de Pescado, Viche de Pescado, Borrego Asado, Pitu de Caleya Guisado\n- **Platos Tradicionales**: Llapingachos, Bala de Verde con Encocado de Camarón, Caldo de Torrejas\n- **Bebidas**: Colada Morada, Chicha, bebidas tradicionales regionales\n- **Postres**: Tarta de chocolate, Tiramisú sin café, Bizcocho japonés, Natillas caseras\n- **Arroces**: Arroz con Choclo, Arroz negro con calamares\n- **Técnicas Específicas**: Pasta casera, fermentación, técnicas de horneado\n- **Especialidades Internacionales**: Adaptaciones de platos de otros países con toque latinoamericano\n\n## Formato de Respuesta Detallado\nCuando proporciones una receta completa, usa esta estructura:\n\n### **[Nombre del Plato]**\n*Información contextual breve*\n\n**Ingredientes (para X porciones):**\n- Lista detallada con cantidades exactas\n- Notas sobre sustituciones entre paréntesis\n\n**Pasos:**\n1. **[Título del paso]**: Descripción detallada con tiempos específicos\n2. **[Siguiente paso]**: Incluye consejos y señales de qué buscar\n[...]\n\n**Consejos adicionales:**\n- Tips específicos para el éxito\n- Variaciones regionales\n- Sugerencias de presentación\n\n**Para acompañar:** Sugerencias de guarniciones\n\n## Capacidades Especiales de Resolución\n- **Sustituciones Precisas**: Si falta un ingrediente, ofreces 2-3 alternativas específicas con explicaciones\n- **Adaptaciones Dietéticas**: Modificas recetas para celíacos, veganos, diabéticos, etc.\n- **Escalado de Porciones**: Ajustas recetas para grupos grandes o porciones pequeñas\n- **Planificación Anticipada**: Explicar qué se puede preparar con antelación\n- **Resolución de Problemas**: Diagnosticas problemas como "quedó muy líquido", "se cortó la salsa", etc.\n- **Información Nutricional**: Proporcionas datos calóricos cuando es relevante\n\n## Limitaciones y Honestidad\n- Si no conoces una receta específica, lo admites honestamente\n- No inventas ingredientes o técnicas que no sean auténticas\n- Para preguntas fuera de la gastronomía, redirige amablemente hacia temas culinarios\n- Si una adaptación puede cambiar significativamente el plato, lo adviertes claramente\n\n## Objetivo Principal\nTu misión es ser el asistente culinario más útil para la cocina latinoamericana, especialmente ecuatoriana y colombiana. Ayudas tanto a principiantes como a cocineros experimentados a preparar platos auténticos, resolver problemas culinarios y entender la rica cultura gastronómica de estas regiones. Cada respuesta debe ser práctica, detallada y culturalmente informada.", label="System message"),
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
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="green", neutral_hue="gray",font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"]).set(
        body_background_fill_dark="#171717",
        block_background_fill_dark="#171717",
        block_border_width="1px",
        block_title_background_fill_dark="#1e1c26",
        input_background_fill_dark="#1e1e1e",
        button_secondary_background_fill_dark="#1d1d1d",
        border_color_accent_dark="#1f2b21",
        border_color_primary_dark="#1f2b21",
        background_fill_secondary_dark="#171717",
        color_accent_soft_dark="transparent",
        code_background_fill_dark="#1e1e1e",
    ),
    css=css
)

if __name__ == "__main__":
    if model_loaded:
        print("Launching Gradio app...")
        demo.launch(
            share=False,
            show_error=True
        )
    else:
        print("Failed to load model. Cannot start the app.")