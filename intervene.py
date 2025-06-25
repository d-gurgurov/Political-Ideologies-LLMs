import pickle
import numpy as np
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from baukit import TraceDict
from einops import rearrange
from collections import defaultdict, Counter
import random
import json
import os

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def layer_head_to_flattened_idx(layer, head, num_heads):
    """Convert (layer, head) tuple to flattened index."""
    return layer * num_heads + head

def flattened_idx_to_layer_head(idx, num_heads):
    """Convert flattened index to (layer, head) tuple."""
    return idx // num_heads, idx % num_heads

def load_probe_bundle(filepath):
    """Load probes and related data from pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded probe bundle from {filepath}")
    return data["probes"], data["label2id"], data["accs"]

def get_top_k_heads_by_accuracy(accs, k, num_heads):
    """
    Returns a list of (layer, head) tuples corresponding to the top-k probe accuracies.
    """
    head_indices = np.argsort(accs)[-k:][::-1]  # highest first
    top_heads = [(idx // num_heads, idx % num_heads) for idx in head_indices]
    return top_heads

def get_head_wise_activations_bau(model, input_ids, device):
    """
    Collect activations from output projection of each self-attention layer.
    Returns: numpy array of shape (num_layers, num_heads, head_dim)
    """
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_dim // num_heads

    HOOKS = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]

    with torch.no_grad():
        input_ids = input_ids.to(device)
        with TraceDict(model, HOOKS) as ret:
            _ = model(input_ids, output_hidden_states=False)

        all_layer_outputs = []
        for hook in HOOKS:
            out = ret[hook].output.squeeze(0).detach().cpu()  # (seq_len, hidden_dim)
            last_token = out[-1]
            per_head = last_token.view(num_heads, head_dim)
            all_layer_outputs.append(per_head)

        all_layer_outputs = torch.stack(all_layer_outputs, dim=0).numpy()  # (num_layers, num_heads, head_dim)
    return all_layer_outputs

def collect_all_activations(model, input_ids_list, device):
    """
    Collect activations for all samples in the dataset.
    Returns: numpy array of shape (num_samples, num_layers, num_heads, head_dim)
    """
    all_activations = []
    for input_ids in tqdm(input_ids_list, desc="Collecting activations"):
        activations = get_head_wise_activations_bau(model, input_ids, device)
        all_activations.append(activations)
    return np.array(all_activations)  # shape: (num_samples, num_layers, num_heads, head_dim)

def compute_center_of_mass_directions(activations, labels, num_classes):
    """
    Compute center of mass (mean activation) for each class.
    Returns: dict mapping class_idx -> numpy array of center of mass vectors
    """
    num_samples, num_layers, num_heads, head_dim = activations.shape
    com_directions = {}
    
    for class_idx in range(num_classes):
        class_indices = np.where(np.array(labels) == class_idx)[0]
        class_activations = activations[class_indices]
        
        for layer in range(num_layers):
            for head in range(num_heads):
                idx = layer_head_to_flattened_idx(layer, head, num_heads)
                if idx not in com_directions:
                    com_directions[idx] = {}
                
                # Compute mean activation for this class in this head
                mean_activation = np.mean(class_activations[:, layer, head], axis=0)
                norm = np.linalg.norm(mean_activation)
                if norm > 0:
                    com_directions[idx][class_idx] = mean_activation / norm
                else:
                    com_directions[idx][class_idx] = np.zeros_like(mean_activation)
    
    return com_directions

def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass=False, 
                          use_random_dir=False, com_directions=None, target_class=None):
    """
    Prepare interventions dictionary with direction vectors and scaling factors.
    """
    interventions = {}
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    
    for layer, head in top_heads:
        idx = layer_head_to_flattened_idx(layer, head, num_heads)
        
        # Select direction vector based on strategy
        if use_center_of_mass:
            print("Using COM")
            if target_class is None:
                raise ValueError("Target class must be specified when using center of mass directions")
            direction = com_directions[idx][target_class]
        elif use_random_dir:
            print("Using random direction")
            head_dim = tuning_activations.shape[-1]
            direction = np.random.normal(size=(head_dim,))
        else:
            # Use probe weight vector as direction
            print("Using probe weights")
            direction = probes[idx].coef_
        
        # Normalize direction vector
        direction = direction / np.linalg.norm(direction)
        
        # Compute projections and standard deviation
        activations = tuning_activations[:, layer, head, :]  # (batch_size, head_dim)
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze(), proj_val_std))
    
    # Sort intervention tuples by head index
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(
            interventions[f"model.layers.{layer}.self_attn.o_proj"], 
            key=lambda x: x[0]
        )
    
    return interventions

def make_modulated_intervention_hook(interventions, alpha, num_heads, edit_start_location='last'):
    """
    Create hook function that adds scaled direction vectors to attention outputs.
    """
    def hook_fn(value, hook_name):
        if hook_name not in interventions:
            return value
            
        value = value.clone()
        # Reshape to separate heads
        value_reshaped = rearrange(value, 'b s (h d) -> b s h d', h=num_heads)
        
        for head_idx, direction, proj_val_std in interventions[hook_name]:
            direction_tensor = torch.tensor(direction, device=value.device, dtype=value.dtype)
            
            # Apply intervention at the appropriate location(s)
            if edit_start_location == 'last':
                value_reshaped[:, -1, head_idx, :] += alpha * proj_val_std * direction_tensor
            else:
                # Edit from the specified position to the end
                start_idx = edit_start_location if isinstance(edit_start_location, int) else 0
                value_reshaped[:, start_idx:, head_idx, :] += alpha * proj_val_std * direction_tensor
        
        # Reshape back to original shape
        value = rearrange(value_reshaped, 'b s h d -> b s (h d)')
        return value
    
    return hook_fn

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

def load_paraphrase_prompts(prompts_dir):
    """Load all paraphrase prompt files (par0.json to par10.json)"""
    prompt_files = []
    prompts_data = {}
    
    for i in range(11):  # 0 to 10
        prompt_file = os.path.join(prompts_dir, f"par{i}.json")
        if os.path.exists(prompt_file):
            prompt_files.append(prompt_file)
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts_data[f"par{i}"] = json.load(f)
            print(f"Loaded paraphrase file: {prompt_file}")
        else:
            print(f"Warning: Paraphrase file not found: {prompt_file}")
    
    return prompts_data

def load_language_questions(questions_dir, lang_code="en"):
    """Load questions for a specific language"""
    lang_file = os.path.join(questions_dir, f"{lang_code}.json")
    if not os.path.exists(lang_file):
        print(f"Warning: No question file for language '{lang_code}' at {lang_file}")
        return []
    
    print(f"Processing language: {lang_code}")
    with open(lang_file, "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    
    questions_by_page = lang_data.get("questions", {})
    flat_questions = [q for page in questions_by_page.values() for q in page]
    
    return flat_questions

def generate_with_intervention(model, tokenizer, messages, interventions, alpha, num_heads, 
                                    device, max_new_tokens=20, edit_start_location='last', stop_tokens=None):
    """
    Generate text with intervention using chat-style prompts.
    Returns only the newly generated text without the input prompt.
    
    Args:
        stop_tokens: List of strings or token IDs to stop generation at
    """
    model.eval()

    # Prepare input using chat template
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)
    original_length = input_ids.shape[1]  # Store original input length
    generated_ids = input_ids.clone()

    hook_names = list(interventions.keys())
    hook_fn = make_modulated_intervention_hook(interventions, alpha, num_heads, edit_start_location)
    edit_output = {hook: hook_fn for hook in hook_names}

    # Prepare stop token IDs
    stop_token_ids = []
    if stop_tokens:
        for stop_token in stop_tokens:
            if isinstance(stop_token, str):
                # Convert string to token ID
                token_ids = tokenizer.encode(stop_token, add_special_tokens=False)
                stop_token_ids.extend(token_ids)
            elif isinstance(stop_token, int):
                stop_token_ids.append(stop_token)
    
    # Always include EOS token as a stop token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        stop_token_ids.append(eos_token_id)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            with TraceDict(model, hook_names, edit_output=edit_output) as _:
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check if we should stop generation
            # if next_token.item() in stop_token_ids:
            #     break

    # Extract only the newly generated tokens (excluding the input prompt)
    new_tokens = generated_ids[0, original_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def process_paraphrases_with_intervention(model, tokenizer, prompts_data, questions, 
                                        interventions, alpha, num_heads, device, 
                                        output_dir, max_new_tokens=100):
    """
    Process all paraphrases with both baseline and intervention for each question.
    Modified to extract only generated responses without prompts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for paraphrase_name, prompt_data in prompts_data.items():
        print(f"\nProcessing paraphrase: {paraphrase_name}")
        
        # Extract prompt prefix from the paraphrase data
        prompt_prefix = prompt_data.get("en", "")
        print(prompt_prefix)
        if not prompt_prefix:
            print(f"Warning: No prompt_prefix found in {paraphrase_name}")
            continue
        
        paraphrase_results = []
        
        for q_idx, question in enumerate(tqdm(questions, desc=f"Processing {paraphrase_name}")):
            # Combine question with prompt prefix
            full_content = f"{question}\n{prompt_prefix}"
            
            # Create chat template
            messages = [
                {"role": "user", "content": full_content}
            ]
            
            try:
                # Check if model is Qwen to conditionally set enable_thinking
                chat_template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                
                # Only add enable_thinking=False for Qwen models
                model_name = model.config.name_or_path if hasattr(model.config, 'name_or_path') else str(model)
                if "qwen" in model_name.lower():
                    chat_template_kwargs["enable_thinking"] = False
                
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    **chat_template_kwargs
                )
            except Exception as e:
                print(f"Warning: Chat template failed, falling back to plain text: {e}")
                formatted_prompt = full_content
            
            # BASELINE GENERATION - Extract only new tokens
            with torch.no_grad():
                input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)
                original_length = input_ids.shape[1]  # Store original input length
                
                baseline_outputs = model.generate(
                    input_ids=input_ids, 
                    max_new_tokens=max_new_tokens, 
                    eos_token_id=None,
                    do_sample=False,  # For reproducibility
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Extract only the newly generated tokens (excluding the input prompt)
                new_tokens = baseline_outputs[0, original_length:]
                baseline_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # INTERVENTION GENERATION - Already modified to return only new text
            intervention_text = generate_with_intervention(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                interventions=interventions,
                alpha=alpha,
                num_heads=num_heads,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            
            # Store results
            result = {
                "question_index": q_idx,
                "question": question,
                "prompt_prefix": prompt_prefix,
                "formatted_prompt": formatted_prompt,
                "baseline_response": baseline_text,
                "intervention_response": intervention_text,
                "paraphrase": paraphrase_name
            }
            
            paraphrase_results.append(result)
        
        # Save results for this paraphrase
        os.makedirs(os.path.join(output_dir, paraphrase_name), exist_ok=True)
        paraphrase_output_path = os.path.join(output_dir, f"{paraphrase_name}/en_{paraphrase_name}_responses.json")
        with open(paraphrase_output_path, "w", encoding="utf-8") as f:
            json.dump(paraphrase_results, f, ensure_ascii=False, indent=2)
        
        print(f"Results for {paraphrase_name} saved to {paraphrase_output_path}")
        all_results.extend(paraphrase_results)
    
    # Save combined results
    combined_output_path = os.path.join(output_dir, "en_all_paraphrases_combined.json")
    with open(combined_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nCombined results saved to {combined_output_path}")
    return all_results

# Example usage
if __name__ == "__main__":
    import os
    import json
    from datetime import datetime
    
    # Set seed if provided
    print(f"Setting seed to {42}")
    set_seed(42)

    # Base Configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" 
    PROBE_PATH = f"probes/probes_bundle_binary_Llama-3.1-8B-Instruct_2000.pkl"
    PROMPTS_DIR = "prompts"  # Directory containing par0.json to par10.json
    QUESTIONS_DIR = "data"  # Directory containing language question files
    MAX_NEW_TOKENS = 100  # Max tokens to generate
    
    # Hyperparameter grids to explore
    K_VALUES = [128, 256, 512]  # Number of heads to intervene on
    ALPHA_VALUES = [20, 25, 30]  # Intervention strength
    TARGET_CLASSES = [1, 0]  # Target classes to try
    
    # Create main results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MAIN_OUTPUT_DIR = f"intervention"
    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)
    
    # Store all results for analysis
    all_results = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer once (expensive operations)
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    
    # Set pad token if needed
    if "llama" in MODEL_NAME.lower():
        tokenizer.pad_token = tokenizer.eos_token
        print("Added pad token")
    
    # Load probes and activations once
    print("Loading probes...")
    probes, label2id, accs = load_probe_bundle(PROBE_PATH)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # Load datasets once
    print("Loading tuning dataset...")
    input_ids_list, label_list, label2id = load_language_id_dataset(tokenizer, max_samples=100, binary=True)
    
    print("Collecting activations...")
    tuning_activations = collect_all_activations(model, input_ids_list, device)
    
    print("Loading paraphrase prompts...")
    prompts_data = load_paraphrase_prompts(PROMPTS_DIR)
    
    print("Loading English questions...")
    questions = load_language_questions(QUESTIONS_DIR, lang_code="en")
    
    if not questions:
        print("No questions found for English. Exiting.")
        exit(1)
    
    print(f"Found {len(questions)} questions for English")
    
    # Compute center of mass directions once
    com = compute_center_of_mass_directions(tuning_activations, label_list, len(label2id))
    
    # Grid search over hyperparameters
    total_runs = len(K_VALUES) * len(ALPHA_VALUES) * len(TARGET_CLASSES)
    current_run = 0
    
    print(f"\nStarting hyperparameter search with {total_runs} total runs...")
    print(f"K values: {K_VALUES}")
    print(f"Alpha values: {ALPHA_VALUES}")
    print(f"Target classes: {TARGET_CLASSES}")
    print("-" * 80)
    
    for k in K_VALUES:
        for alpha in ALPHA_VALUES:
            for target_class in TARGET_CLASSES:
                current_run += 1
                print(f"\nRun {current_run}/{total_runs}: K={k}, Alpha={alpha}, Target={target_class}")
                
                # Create output directory for this run
                run_dir = f"k{k}_alpha{alpha}_target{target_class}"
                output_dir = os.path.join(MAIN_OUTPUT_DIR, run_dir)
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    # Get top heads for this K value
                    top_heads = get_top_k_heads_by_accuracy(accs, k, num_heads)
                    print(f"  Using top {k} heads")
                    
                    # Get interventions for this configuration
                    interventions = get_interventions_dict(
                        top_heads=top_heads,
                        probes=probes,
                        tuning_activations=tuning_activations,
                        num_heads=num_heads,
                        use_center_of_mass=True,
                        com_directions=com,
                        use_random_dir=False,
                        target_class=target_class
                    )
                    
                    # Run the experiment
                    print(f"  Processing paraphrases...")
                    results = process_paraphrases_with_intervention(
                        model=model,
                        tokenizer=tokenizer,
                        prompts_data=prompts_data,
                        questions=questions,
                        interventions=interventions,
                        alpha=alpha,
                        num_heads=num_heads,
                        device=device,
                        output_dir=output_dir,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    
                    # Store results with hyperparameters
                    run_summary = {
                        'k': k,
                        'alpha': alpha,
                        'target_class': target_class,
                        'output_dir': output_dir,
                        'num_results': len(results),
                        'status': 'success',
                        'top_heads_sample': top_heads[:5].tolist() if len(top_heads) > 0 else []
                    }
                    
                    # Add any metrics you want to track from results
                    if results:
                        # Example: track average response length or other metrics
                        avg_response_length = np.mean([len(r.get('response', '')) for r in results])
                        run_summary['avg_response_length'] = avg_response_length
                    
                    all_results.append(run_summary)
                    
                    print(f"  ✓ Completed: {len(results)} results saved to {output_dir}")
                    
                except Exception as e:
                    print(f"  ✗ Error in run K={k}, Alpha={alpha}, Target={target_class}: {str(e)}")
                    error_summary = {
                        'k': k,
                        'alpha': alpha,
                        'target_class': target_class,
                        'output_dir': output_dir,
                        'status': 'error',
                        'error': str(e)
                    }
                    all_results.append(error_summary)
                    continue

    