import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model from local path
local_model_path = "/home/uom/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Sample small prompt set (can replace with your real prompts)
instruction_prompts = [
    "Translate the following English sentence to French: 'Hello, how are you?'",
    "Summarize the following text in one sentence.",
    "Write a short story about a dragon who loves cooking.",
    "List three benefits of daily exercise.",
    "Explain the importance of recycling.",
    "Describe the process of photosynthesis.",
    "What are the key features of Python programming language?",
    "Provide a recipe for making pancakes.",
    "Create a poem about the ocean.",
    "Write a letter applying for a job."
]

reasoning_prompts = [
    "If a train travels at 60 mph for 2 hours, how far does it go?",
    "What is the next number in the sequence: 2, 4, 8, 16?",
    "Solve for x: 2x + 3 = 11.",
    "If a rectangle has length 10 and width 5, what is the area?",
    "If a = 5 and b = 3, what is the value of a^2 + b^2?",
    "You have 3 red balls and 2 blue balls. What is the probability of picking a red ball?",
    "Is 17 a prime number? Explain why.",
    "What is the square root of 144?",
    "If it is raining, the ground is wet. The ground is wet. Is it necessarily raining?",
    "There are 5 apples and 3 kids. How many apples does each get if divided equally?"
]

# Labels: 1 = instruction, 0 = reasoning
all_prompts = instruction_prompts + reasoning_prompts
y = np.array([1] * len(instruction_prompts) + [0] * len(reasoning_prompts))

# Choose layer index
layer_idx = 12

# Get hidden states from layer 12
X = []
for prompt in all_prompts:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[layer_idx]  # Get only layer 12
        pooled = hidden_state.mean(dim=1).squeeze().cpu()
        X.append(pooled)

X = torch.stack(X).numpy()

# Shuffle data
X, y = shuffle(X, y, random_state=42)

# Train/test split (balanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Logistic regression with regularization
clf = LogisticRegression(max_iter=1000, C=0.01)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
instr_acc = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
reason_acc = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])

print(f"Layer {layer_idx} Accuracy: {acc:.4f}")
print(f"Instruction Accuracy: {instr_acc:.4f}")
print(f"Reasoning Accuracy: {reason_acc:.4f}")
