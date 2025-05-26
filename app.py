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

# Custom CSS for better appearance
css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.header {
    text-align: center;
    margin-bottom: 20px;
}

.examples {
    margin: 20px 0;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, title="🍽️ Chef Virtual Ecuatoriano-Colombiano") as demo:
    
    gr.HTML("""
    <div class="header">
        <h1>🍽️ Chef Virtual: Patrimonio Gastronómico Ecuatoriano-Colombiano</h1>
        <p>¡Descubre los sabores tradicionales de Ecuador y Colombia! Pregúntame sobre recetas, ingredientes y técnicas culinarias.</p>
    </div>
    """)
    
    # Main chat interface
    chatbot = gr.ChatInterface(
        fn=generate,
        chatbot=gr.Chatbot(
            height=500,
            placeholder="¡Hola! Soy tu chef virtual especializado en gastronomía ecuatoriana y colombiana. ¿En qué puedo ayudarte hoy?"
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
            ["¿Cómo se prepara la arepa colombiana tradicional?"],
            ["Dame una receta completa de sancocho de gallina criolla"],
            ["¿Qué diferencias hay entre el ceviche ecuatoriano y el peruano?"],
            ["¿Cómo hacer empanadas de verde ecuatorianas?"],
            ["Receta de bandeja paisa colombiana paso a paso"],
            ["¿Qué postres típicos puedo hacer con panela?"],
            ["Ingredientes y preparación del encebollado ecuatoriano"],
            ["¿Cómo se hace el chocolate santafereño?"],
            ["Receta de humitas ecuatorianas dulces"],
        ],
        cache_examples=False,
        retry_btn="🔄 Reintentar",
        undo_btn="↩️ Deshacer",
        clear_btn="🗑️ Limpiar conversación",
        submit_btn="📤 Enviar",
        stop_btn="⏹️ Detener",
    )
    
    # Footer with information
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 10px;">
        <p><strong>Modelo:</strong> Mistral 7B fine-tuneado en patrimonio gastronómico ecuatoriano-colombiano</p>
        <p><strong>Datos:</strong> Recetas tradicionales, técnicas culinarias y conocimiento gastronómico regional</p>
        <p><em>🔥 Powered by ZeroGPU • 🤗 Hugging Face Spaces</em></p>
    </div>
    """)

if __name__ == "__main__":
    if model_loaded:
        print("🚀 Launching Gradio app...")
        demo.launch(
            share=False,
            show_error=True,
            debug=True
        )
    else:
        print("❌ Failed to load model. Cannot start the app.")