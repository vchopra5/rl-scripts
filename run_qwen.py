import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
import random

model_path = "./models/Qwen3.5-0.8B-Base"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForMultimodalLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

random.seed(42)

def run_model(n1, n2, preamble=True):
    global model
    
    model.eval()

    if preamble:
        prompt = "2 + 3 = 5\n7 + 1 = 8\n"
    else:
        prompt = ""

    if random.random() < 0.5:
        prompt += f"{n1} + {n2} ="
    else:
        prompt += f"{n2} + {n1} ="

    inputs = processor(text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            num_return_sequences=8,
            pad_token_id = processor.tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    answer_tokens = output[:, prompt_length:]
    answers = processor.batch_decode(answer_tokens, skip_special_tokens=True)

    return output, prompt_length, answers
