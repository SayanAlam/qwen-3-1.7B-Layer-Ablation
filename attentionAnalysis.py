import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Qwen3-1.7B model and tokenizer from local path
local_model_path = "/home/uom/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)
model.eval().to("cuda")  # Use GPU

# Reasoning and instruction prompts
reasoning_prompts = [
    "Every morning, Aya goes for a 9-kilometer-long walk and then stops at a coffee shop afterward. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks at a speed of s + 2 kilometers per hour, the walk takes her 2 hours and 24 minutes, again including t minutes in the coffee shop. Now, suppose Aya walks at a speed of s + 0.5 kilometers per hour. Find the total number of minutes the walk takes her, including the time spent in the coffee shop.",
]

instruction_prompts = [
    "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.",
]

def compute_attention_focus(prompt_list):
    all_focus = []

    for prompt in prompt_list:
        # Set max_length to 1024 to prevent overflow
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        attentions = outputs.attentions  # List of (batch, heads, seq_len, seq_len)

        # Compute average attention across heads and tokens
        focus_by_layer = []
        for layer_attn in attentions:
            avg_attn = layer_attn[0].mean(dim=0)  # (seq_len, seq_len)
            focus = avg_attn.sum(dim=1).mean().item() / avg_attn.size(0)
            focus_by_layer.append(focus)

        all_focus.append(focus_by_layer)

    # Average across prompts
    avg_focus = torch.tensor(all_focus).mean(dim=0).tolist()
    return avg_focus

# Run analysis
instruction_focus = compute_attention_focus(instruction_prompts)
reasoning_focus = compute_attention_focus(reasoning_prompts)

# Print results
print("\nAttention Focus per Layer (Averaged Over Prompts)")
print("--------------------------------------------------")
print("Layer | Instruction Focus | Reasoning Focus")
print("--------------------------------------------------")

lines = []
lines.append("Layer-wise Attention Focus Table:\n")
lines.append("Layer | Instruction Focus | Reasoning Focus")
lines.append("--------------------------------------------")

for i, (inst, reas) in enumerate(zip(instruction_focus, reasoning_focus)):
    line = f"{i:5} |       {inst:.6f}     |     {reas:.6f}"
    print(line)
    lines.append(line)

# Save results to a .txt file
output_path = "attentionAnalysis.txt"
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"\nResults saved to {output_path}")

