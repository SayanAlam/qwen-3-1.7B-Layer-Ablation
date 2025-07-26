import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def get_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model.eval(), tokenizer

def layer_ablation(model, tokenizer, layer_to_ablate, prompt):
    handle = None

    if layer_to_ablate >= 0:
        try:
            target_block = model.model.layers[layer_to_ablate]
        except AttributeError:
            raise RuntimeError(f"Layer structure not found at index {layer_to_ablate}")

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return tuple(torch.zeros_like(o) if isinstance(o, torch.Tensor) else o for o in output)
            return torch.zeros_like(output)

        handle = target_block.register_forward_hook(hook_fn)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt", return_token_type_ids=False, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True
        )
        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    if handle:
        handle.remove()

    return response.strip()

def cosine_similarity(a, b, model):
    emb1 = model.encode(a, convert_to_tensor=True, device='cuda')
    emb2 = model.encode(b, convert_to_tensor=True, device='cuda')
    return util.pytorch_cos_sim(emb1, emb2).item()

def main():
    model_path = "/home/uom/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"
    model, tokenizer = get_model_and_tokenizer(model_path)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = embedder.to("cuda")

    num_layers = 28

    reasoning_prompts = [
        "Every morning, Aya goes for a 9-kilometer-long walk and then stops at a coffee shop afterward. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks at a speed of s + 2 kilometers per hour, the walk takes her 2 hours and 24 minutes, again including t minutes in the coffee shop. Now, suppose Aya walks at a speed of s + 0.5 kilometers per hour. Find the total number of minutes the walk takes her, including the time spent in the coffee shop.",
    ]

    instruction_prompts = [
        "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.",
    ]

    print("Generating baseline responses (no ablation)...")
    base_reasoning_outputs = [layer_ablation(model, tokenizer, -1, p) for p in reasoning_prompts]
    base_instruction_outputs = [layer_ablation(model, tokenizer, -1, p) for p in instruction_prompts]

    print("\nStarting full-layer ablation with prompt averaging...\n")
    header = f"{'Layer':>5} | {'Avg Δ Reasoning':>16} | {'Avg Δ Instruction':>18}"
    separator = "-" * len(header)

    print(header)
    print(separator)

    output_lines = ["Layer-wise Cosine Similarity Δ Table:\n", header, separator]

    for i in tqdm(range(num_layers), desc="Layer Ablation"):
        reasoning_deltas = []
        instruction_deltas = []

        for idx, prompt in enumerate(reasoning_prompts):
            ablated_output = layer_ablation(model, tokenizer, i, prompt)
            sim = cosine_similarity(base_reasoning_outputs[idx], ablated_output, embedder)
            reasoning_deltas.append(1.0 - sim)

        for idx, prompt in enumerate(instruction_prompts):
            ablated_output = layer_ablation(model, tokenizer, i, prompt)
            sim = cosine_similarity(base_instruction_outputs[idx], ablated_output, embedder)
            instruction_deltas.append(1.0 - sim)

        avg_delta_r = sum(reasoning_deltas) / len(reasoning_deltas)
        avg_delta_i = sum(instruction_deltas) / len(instruction_deltas)

        line = f"{i:>5} | {avg_delta_r:16.4f} | {avg_delta_i:18.4f}"
        print(line)
        output_lines.append(line)

        torch.cuda.empty_cache()
        gc.collect()

    # Save results to a text file
    output_path = "layerAblationAnalysis.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
