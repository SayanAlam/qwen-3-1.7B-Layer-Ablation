import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def layer_ablation(model, tokenizer, layer_to_ablate, prompt):
    handle = None

    if layer_to_ablate >= 0:
        target_block = model.model.layers[layer_to_ablate]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return tuple(torch.zeros_like(o) if isinstance(o, torch.Tensor) else o for o in output)
            return torch.zeros_like(output)

        handle = target_block.register_forward_hook(hook_fn)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(inputs.input_ids, max_new_tokens=1024)
        response = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    if handle:
        handle.remove()

    return response.strip()

def cosine_similarity(a, b, model):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def main():
    model_name = "Qwen/Qwen3-1.7B"
    model, tokenizer = get_model_and_tokenizer(model_name)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    num_layers = 28

    # 10 diverse reasoning prompts
    reasoning_prompts = [
        "Every morning, Aya goes for a 9-kilometer-long walk and then stops at a coffee shop afterward. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks at a speed of s + 2 kilometers per hour, the walk takes her 2 hours and 24 minutes, again including t minutes in the coffee shop. Now, suppose Aya walks at a speed of s + 0.5 kilometers per hour. Find the total number of minutes the walk takes her, including the time spent in the coffee shop.",
        "There exist real numbers x and y, both greater than 1, such that logₓ(yˣ) = logᵧ(x⁴ʸ) = 10. Find the value of xy.",
        "Alice and Bob play the following game. A stack of n tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. Find the number of positive integers n less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play.",
        "Jen enters a lottery by picking 4 distinct numbers from S = {1, 2, 3, ..., 10}. Four numbers are randomly chosen from S. She wins a prize if at least two of her numbers were two of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is m/n, where m and n are relatively prime positive integers. Find m + n.",
        "Rectangles ABCD and EFGH are drawn such that D, E, C, and F are collinear. Also, A, D, H, and G all lie on a circle. If BC = 16, AB = 107, FG = 17, and EF = 184, what is the length of CE?",
        "Consider the paths of length 16 that follow the lines from the lower left corner to the upper right corner on an 8 x 8 grid. Find the number of such paths that change direction exactly four times.",
        "Eight circles of radius 34 are sequentially tangent, and two of the circles are tangent to AB and BC of triangle ABC, respectively. 2024 circles of radius 1 can be arranged in the same manner. The inradius of triangle ABC can be expressed as m/n, where m and n are relatively prime positive integers. Find m + n.",
        "Let triangle ABC have side lengths AB = 5, BC = 9, and CA = 10. The tangents to the circumcircle of triangle ABC at points B and C intersect at point D, and line segment AD intersects the circumcircle again at point P (other than A). The length of AP is equal to m/n, where m and n are relatively prime integers. Find m + n.",
        "Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability that the octagon can then be rotated so that all of the blue vertices end up at positions where there had been red vertices is m/n, where m and n are relatively prime positive integers. Find m + n.",
        "Let p be the least prime number for which there exists a positive integer n such that n^4 + 1 is divisible by p^2. Find the least positive integer m such that m^4 + 1 is divisible by p^2."
    ]

    # 10 diverse instruction-following prompts
    instruction_prompts = [
        "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.",
        "Write a resume for a fresh high school graduate who is seeking their first job. Make sure to include at least 12 placeholder represented by square brackets, such as [address], [name].",
        "Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. <<title>>.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)'",
        "Given the sentence \"Two young boys with toy guns and horns.\" can you ask a question? Please ensure that your response is in English, and in all lowercase letters. No capital letters are allowed.",
        "Write a dialogue between two people, one is dressed up in a ball gown and the other is dressed down in sweats. The two are going to a nightly event. Your answer must contain exactly 3 bullet points in the markdown format (use \"* \" to indicate each bullet) such as:\n* This is the first point.\n* This is the second point.",
        "Write a 2 paragraph critique of the following sentence in all capital letters, no lowercase letters allowed: \"If the law is bad, you should not follow it\". Label each paragraph with PARAGRAPH X.",
        "Write me a resume for Matthias Algiers. Use words with all capital letters to highlight key abilities, but make sure that words with all capital letters appear less than 10 times. Wrap the entire response with double quotation marks.",
        "Write a letter to a friend in all lowercase letters ask them to go and vote.",
        "Write a long email template that invites a group of participants to a meeting, with at least 500 words. The email must include the keywords \"correlated\" and \"experiencing\" and should not use any commas.",
        "Write a story of exactly 2 paragraphs about a man who wakes up one day and realizes that he's inside a video game. Separate the paragraphs with the markdown divider: ***"
    ]

    print("🔹 Generating baseline responses (no ablation)...")
    base_reasoning_outputs = [layer_ablation(model, tokenizer, -1, p) for p in reasoning_prompts]
    base_instruction_outputs = [layer_ablation(model, tokenizer, -1, p) for p in instruction_prompts]

    print("\n🔸 Starting full-layer ablation with 10-prompt averaging...\n")
    print(f"{'Layer':>5} | {'Avg Δ Reasoning':>16} | {'Avg Δ Instruction':>18}")
    print("-" * 47)

    for i in tqdm(range(num_layers), desc="Layer Ablation"):
        reasoning_deltas = []
        instruction_deltas = []

        for idx, prompt in enumerate(reasoning_prompts):
            ablated_output = layer_ablation(model, tokenizer, i, prompt)
            sim = cosine_similarity(base_reasoning_outputs[idx], ablated_output, embedder)
            delta = 1.0 - sim
            reasoning_deltas.append(delta)

        for idx, prompt in enumerate(instruction_prompts):
            ablated_output = layer_ablation(model, tokenizer, i, prompt)
            sim = cosine_similarity(base_instruction_outputs[idx], ablated_output, embedder)
            delta = 1.0 - sim
            instruction_deltas.append(delta)

        avg_delta_r = sum(reasoning_deltas) / len(reasoning_deltas)
        avg_delta_i = sum(instruction_deltas) / len(instruction_deltas)

        print(f"{i:>5} | {avg_delta_r:16.4f} | {avg_delta_i:18.4f}")

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
