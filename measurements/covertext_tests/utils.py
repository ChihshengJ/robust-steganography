import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name="gpt2-large", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


def compute_perplexity(text, model, tokenizer, device, max_length=1024):
    encodings = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = encodings.input_ids.to(device)

    if input_ids.shape[1] < 2:
        return float("nan")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss

    return torch.exp(neg_log_likelihood).item()


def compute_perplexities(texts, model_name="gpt2-large", device=None, max_length=1024):
    model, tokenizer, device = load_model(model_name, device)
    results = []
    for text in texts:
        ppl = compute_perplexity(text, model, tokenizer, device, max_length)
        results.append(ppl)
    return results
