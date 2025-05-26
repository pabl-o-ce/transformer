import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables
model = None
tokenizer = None

def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer
    
    model_name = "somosnlp-hackathon-2025/mistral-7B-ec-es-recetas"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_name} on {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        print("✅ Model loaded successfully!")
        return "✅ ¡Modelo cargado! Ya puedes hacer preguntas."
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"❌ Error cargando modelo: {str(e)}"

@spaces.GPU(duration=120)
def chat_fn(message, history):
    """Simple chat function"""
    global model, tokenizer
    
    if model is None:
        return "❌ Primero carga el modelo presionando el botón de arriba."
    
    # Format conversation
    conversation = ""
    for user_msg, bot_msg in history:
        conversation += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if bot_msg:
            conversation += f"<|im_start|>assistant\n{bot_msg}<|im_end|>\n"
    
    conversation += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate response
    try:
        inputs = tokenizer.encode(conversation, return_tensors="pt", truncation=True, max_length=1500)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Clean response
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        return response.strip()
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="🍽️ Chat de Recetas") as demo:
    
    gr.Markdown("# 🍽️ Chat de Recetas Ecuatorianas y Colombianas")
    
    load_btn = gr.Button("🚀 Cargar Modelo", variant="primary")
    status = gr.Textbox(label="Estado", value="Presiona 'Cargar Modelo' para comenzar")
    
    chatbot = gr.ChatInterface(
        fn=chat_fn,
        title="Chatea sobre recetas",
        description="Pregúntame sobre recetas ecuatorianas y colombianas",
        examples=[
            "¿Cómo hacer locro ecuatoriano?",
            "Receta de empanadas colombianas",
            "¿Qué ingredientes lleva el sancocho?",
        ],
        retry_btn=None,
        undo_btn=None,
        clear_btn="🗑️ Limpiar"
    )
    
    load_btn.click(load_model, outputs=status)

if __name__ == "__main__":
    demo.launch(debug=True, share=False)