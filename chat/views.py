from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from huggingface_hub import login, HfApi
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# Add your Hugging Face token here
login(token="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", add_to_git_credential=True)

# List of ChatGPT-4 alternatives
models = [
    
    "gpt2",
    "EleutherAI/gpt-neo-125m",
    "huawei-noah/TinyBERT_General_4L_312D",
    "albert/albert-base-v2",
    "openai-community/gpt2-medium",
]

# Load models and tokenizers
loaded_models = {}
for model_name in models:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        loaded_models[model_name] = (tokenizer, model)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

def index(request):
    api = HfApi()
    models_info = []
    
    return render(request, 'chat/index.html', {'models_info': models_info})

def generate_response(model_name, tokenizer, model, user_input):
    try:
        start_time = time.time()
        inputs = tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed_time = time.time() - start_time
        return model_name, {
            'response': bot_response,
            'time': f"{elapsed_time:.2f} seconds"
        }
    except Exception as e:
        print(f"Error generating response for {model_name}: {str(e)}")
        return model_name, {
            'response': f'Error: {str(e)}',
            'time': 'N/A'
        }

@csrf_exempt
def get_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        responses = {}

        with ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(generate_response, model_name, tokenizer, model, user_input): model_name 
                               for model_name, (tokenizer, model) in loaded_models.items()}
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model_name, result = future.result(timeout=300)  # 5 minutes timeout
                    responses[model_name] = result
                except TimeoutError:
                    responses[model_name] = {'response': 'Model timed out', 'time': 'N/A'}
                except Exception as e:
                    responses[model_name] = {'response': f'Error: {str(e)}', 'time': 'N/A'}

        api = HfApi()
        models_info = [{'name': model_name} for model_name in models]

        return render(request, 'chat/index.html', {'responses': responses, 'models_info': models_info})

    return render(request, 'chat/index.html')
