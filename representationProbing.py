import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model and tokenizer from local path
model_name = "/home/uom/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
model.eval().to(device)

# Prompts
reasoning_prompts = [
    "Every morning, Aya goes for a 9-kilometer-long walk and then stops at a coffee shop afterward. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks at a speed of s + 2 kilometers per hour, the walk takes her 2 hours and 24 minutes, again including t minutes in the coffee shop. Now, suppose Aya walks at a speed of s + 0.5 kilometers per hour. Find the total number of minutes the walk takes her, including the time spent in the coffee shop.",
]

instruction_prompts = [
    "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.",
]

all_prompts = instruction_prompts + reasoning_prompts
instruction_labels = [1] * len(instruction_prompts) + [0] * len(reasoning_prompts)
reasoning_labels = [0] * len(instruction_prompts) + [1] * len(reasoning_prompts)

# Feature extraction
def extract_features(prompts, batch_size=5, use_mean_pool=True):
    X_layers = None

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Token length limit enforced here
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)
            hidden_states = outputs.hidden_states[1:]

        attention_mask = tokens["attention_mask"].unsqueeze(-1)

        batch_features = []
        for layer in hidden_states:
            if use_mean_pool:
                masked = layer * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                pooled = summed / counts
            else:
                pooled = layer[:, -1, :]
            batch_features.append(pooled.cpu())

        if X_layers is None:
            X_layers = [layer_chunk.clone().detach() for layer_chunk in batch_features]
        else:
            for j in range(len(X_layers)):
                X_layers[j] = torch.cat((X_layers[j], batch_features[j]), dim=0)

    return X_layers

# Extract features
X_layers = extract_features(all_prompts)

# Train and evaluate linear probes layer-wise
table_data = {
    "Layer": [],
    "Instruction Accuracy": [],
    "Reasoning Accuracy": []
}

for i, X in enumerate(X_layers):
    X = X.numpy()
    X_train, X_test, y_instr_train, y_instr_test = train_test_split(X, instruction_labels, test_size=0.3, random_state=42)
    _, _, y_reasn_train, y_reasn_test = train_test_split(X, reasoning_labels, test_size=0.3, random_state=42)

    clf_instr = LogisticRegression(max_iter=1000)
    clf_reasn = LogisticRegression(max_iter=1000)

    clf_instr.fit(X_train, y_instr_train)
    clf_reasn.fit(X_train, y_reasn_train)

    acc_instr = accuracy_score(y_instr_test, clf_instr.predict(X_test))
    acc_reasn = accuracy_score(y_reasn_test, clf_reasn.predict(X_test))

    table_data["Layer"].append(i)
    table_data["Instruction Accuracy"].append(round(acc_instr, 4))
    table_data["Reasoning Accuracy"].append(round(acc_reasn, 4))

# Show results
df = pd.DataFrame(table_data)
print("\nLayer-wise Accuracy Table:\n")
print(df.to_string(index=False))

# Save results to a .txt file
output_path = "representationProbing.txt"
with open(output_path, "w") as f:
    f.write("Layer-wise Accuracy Table:\n\n")
    f.write(df.to_string(index=False))
print(f"\nResults saved to {output_path}")
