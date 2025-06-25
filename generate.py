import os
import json
import argparse
import random
import torch
import numpy as np
from vllm import LLM, SamplingParams
import glob
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_prompt_file(llm, tokenizer, sampling_params, prompt_file, questions_dir, output_dir, model_name):
    """Process a single prompt file with the already loaded model"""
    print(f"\nProcessing prompt file: {prompt_file}")
    
    # Load multilingual base prompts
    with open(prompt_file, "r", encoding="utf-8") as f:
        base_prompts = json.load(f)
    
    # Create output subdirectory for this prompt file
    prompt_filename = Path(prompt_file).stem  # e.g., "par0" from "par0.json"
    prompt_output_dir = os.path.join(output_dir, prompt_filename)
    if not os.path.exists(prompt_output_dir):
        os.makedirs(prompt_output_dir)
    
    # Loop through each language
    for lang_code, prompt_prefix in base_prompts.items():
        lang_file = os.path.join(questions_dir, f"{lang_code}.json")
        if not os.path.exists(lang_file):
            print(f"Warning: No question file for language '{lang_code}' at {lang_file}")
            continue
        
        print(f"  Processing language: {lang_code}")
        with open(lang_file, "r", encoding="utf-8") as f:
            lang_data = json.load(f)
        
        questions_by_page = lang_data.get("questions", {})
        flat_questions = [q for page in questions_by_page.values() for q in page]
        
        lang_outputs = []
        for q in flat_questions:
            # Combine question with prompt prefix
            full_content = f"{q}\n{prompt_prefix}"
            
            # Create chat template
            messages = [
                {"role": "user", "content": full_content}
            ]
            
            # Apply chat template
            try:
                # Check if model is Qwen to conditionally set enable_thinking
                chat_template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                
                # Only add enable_thinking=False for Qwen models
                if "qwen" in model_name.lower():
                    chat_template_kwargs["enable_thinking"] = False
                
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    **chat_template_kwargs
                )
            except Exception as e:
                print(f"Warning: Chat template failed for {lang_code}, falling back to plain text: {e}")
                formatted_prompt = full_content
            
            # Generate response
            output = llm.generate(prompts=[formatted_prompt], sampling_params=sampling_params)
            answer = output[0].outputs[0].text.strip()
            
            lang_outputs.append({
                "question": q,
                "prompt_prefix": prompt_prefix,
                "formatted_prompt": formatted_prompt,
                "response": answer
            })
        
        # Save this language's outputs to its own file
        lang_output_path = os.path.join(prompt_output_dir, f"{lang_code}_responses.json")
        with open(lang_output_path, "w", encoding="utf-8") as lf:
            json.dump(lang_outputs, lf, ensure_ascii=False, indent=2)
        
        print(f"    Responses for {lang_code} saved to {lang_output_path}")

def main(args):
    # Set seed if provided
    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        set_seed(args.seed)
    
    # Initialize vLLM with the specified model (load once)
    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=4)
    
    # Get tokenizer for chat template
    tokenizer = llm.get_tokenizer()
    
    # Get eos_token_id
    eos_token_id = tokenizer.eos_token_id
    stop_ids = [eos_token_id] if eos_token_id is not None else []
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_ids,
        skip_special_tokens=True,
    )
    
    # Find all prompt files
    if args.prompt_pattern:
        # Use the provided pattern
        prompt_files = glob.glob(args.prompt_pattern)
    else:
        # Default pattern for par0.json to par10.json
        prompt_files = []
        for i in range(11):  # 0 to 10
            prompt_file = os.path.join(args.prompts_dir, f"par{i}.json")
            if os.path.exists(prompt_file):
                prompt_files.append(prompt_file)
    
    if not prompt_files:
        print(f"No prompt files found matching pattern. Check your prompts directory: {args.prompts_dir}")
        return
    
    # Sort files to ensure consistent processing order
    prompt_files.sort()
    
    print(f"Found {len(prompt_files)} prompt files to process:")
    for pf in prompt_files:
        print(f"  - {pf}")
    
    # Process each prompt file sequentially
    for prompt_file in prompt_files:
        try:
            process_prompt_file(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                prompt_file=prompt_file,
                questions_dir=args.questions_dir,
                output_dir=args.output_dir,
                model_name=args.model
            )
        except Exception as e:
            print(f"Error processing {prompt_file}: {e}")
            continue
    
    print(f"\nAll prompt files processed successfully!")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multilingual prompts for multiple prompt files using vLLM")
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model name or path (e.g. mistralai/Mistral-7B-Instruct)")
    parser.add_argument("--prompts_dir", type=str, default="./prompts",
                        help="Directory containing prompt files (par0.json, par1.json, etc.)")
    parser.add_argument("--prompt_pattern", type=str, default=None,
                        help="Custom glob pattern for prompt files (e.g., './prompts/par*.json')")
    parser.add_argument("--questions_dir", type=str, default="data",
                        help="Directory containing language question files (e.g., bg.json, de.json, etc.)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save per-language response JSON files")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)