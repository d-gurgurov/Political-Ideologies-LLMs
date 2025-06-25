from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from baukit import TraceDict
import argparse
from huggingface_hub import login
from collections import defaultdict, Counter

login("*")

import pickle
import os
import random

def save_probe_bundle(probes, label2id, accs, filepath):
    data = {
        "probes": probes,
        "label2id": label2id,
        "accs": accs
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved probe bundle to {filepath}")

def load_language_id_dataset(tokenizer, max_samples=1000, binary=False):
    dataset = load_dataset("JyotiNayak/political_ideologies", split="train")
    input_ids_list = []
    label_list = []
    if binary:
        label2id = {f"conservative": 0, "liberal": 1}
        for i in range(len(dataset)):
            example = dataset[i]
            text = example['statement']
            label = example['label']
            tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            input_ids_list.append(tokenized["input_ids"])
            label_list.append(label)
        # Print stats
        count_pos = sum(label_list)
        count_neg = len(label_list) - count_pos
    else:
        print("The dataset only has two classes!")
    return input_ids_list, label_list, label2id

def get_head_wise_activations_bau(model, input_ids, device):
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_dim // num_heads

    # Hook after o_proj
    HOOKS = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]

    with torch.no_grad():
        input_ids = input_ids.to(device)
        with TraceDict(model, HOOKS) as ret:
            _ = model(input_ids, output_hidden_states=False)

        all_layer_outputs = []
        for hook in HOOKS:
            # (1, seq_len, hidden_dim)
            out = ret[hook].output.squeeze(0).detach().cpu()  # (seq_len, hidden_dim)
            last_token = out[-1]  # use last token only, shape (hidden_dim,)
            per_head = last_token.view(num_heads, head_dim)  # shape: (num_heads, head_dim)
            all_layer_outputs.append(per_head)

        # shape: (num_layers, num_heads, head_dim)
        all_layer_outputs = torch.stack(all_layer_outputs, dim=0).numpy()

    return all_layer_outputs

def collect_all_activations(model, input_ids_list, device):
    all_activations = []
    for input_ids in tqdm(input_ids_list, desc="Collecting activations"):
        activations = get_head_wise_activations_bau(model, input_ids, device)
        activations_avg = activations  # we already take last token only
        all_activations.append(activations_avg)
    return np.array(all_activations)  # shape: (num_samples, num_layers, num_heads, head_dim)


def train_language_id_probes(seed, activations, labels, num_layers, num_heads):
    X_train, X_val, y_train, y_val = train_test_split(
        activations, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    probes = []
    all_head_accs = []

    for layer in tqdm(range(num_layers), desc="Train probes"):
        for head in tqdm(range(num_heads), desc="Current layer"):
            X_train_h = X_train[:, layer, head, :]
            X_val_h = X_val[:, layer, head, :]
            clf = LogisticRegression(random_state=seed, max_iter=1000)
            clf.fit(X_train_h, y_train)
            y_pred_val = clf.predict(X_val_h)
            acc = accuracy_score(y_val, y_pred_val)
            probes.append(clf)
            all_head_accs.append(acc)

    return probes, np.array(all_head_accs)


def visualize_probe_accuracies(accs, num_layers, num_heads, model_name, title="Language ID Probe Accuracy"):
    accs_matrix = accs.reshape((num_layers, num_heads))

    plt.figure(figsize=(num_heads, num_layers))
    sns.heatmap(accs_matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True,
                xticklabels=[f"H{h}" for h in range(num_heads)],
                yticklabels=[f"L{l}" for l in range(num_layers)])
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.title(title)
    plt.tight_layout()
    model_name_safe = model_name.split("/")[-1]
    plt.savefig(f"{model_name_safe}.png")
    plt.show()


def run_language_id_pipeline(model, tokenizer, device, seed=42, max_samples=10000, binary=False):
    input_ids_list, labels, label2id = load_language_id_dataset(tokenizer, max_samples, binary=binary)
    activations = collect_all_activations(model, input_ids_list, device)
    print(activations.shape)
    num_layers, num_heads = activations.shape[1], activations.shape[2]
    probes, accs = train_language_id_probes(seed, activations, labels, num_layers, num_heads)
    return probes, accs, label2id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train language ID probes")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path from HuggingFace")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max number of samples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--binary", action="store_true", help="Enable binary classification mode (one-vs-all)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if "llama" in args.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        print("Added token")

    probes, accs, label2id = run_language_id_pipeline(
        model, tokenizer, device,
        seed=args.seed,
        max_samples=args.max_samples,
        binary=args.binary
    )
    
    # Visualize results
    visualize_probe_accuracies(
        accs, 
        model.config.num_hidden_layers, 
        model.config.num_attention_heads, 
        args.model_name
    )
    
    # Save results
    save_path = os.path.join(
        f"probes_bundle_binary_{args.model_name.split('/')[-1]}_{args.max_samples}.pkl"
    )
    save_probe_bundle(probes, label2id, accs, save_path)
    print(f"Saved binary classification results to {save_path}")

