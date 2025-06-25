import json
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import defaultdict
import argparse
from pathlib import Path

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Analyze political compass baseline vs intervention responses.")
parser.add_argument('--responses_dir', type=str, default='outputs',
                    help='Directory containing paraphrase subdirectories with response JSON files')
parser.add_argument('--data_dir', type=str, default='data',
                    help='Directory containing choice files per language')
parser.add_argument('--output_dir', type=str, default='intervention_analysis',
                    help='Directory to save plots and analysis outputs')
parser.add_argument('--paraphrase_pattern', type=str, default='par*',
                    help='Pattern to match paraphrase directories (e.g., "par*" or "experiment_*")')

args = parser.parse_args()

RESPONSES_DIR = args.responses_dir
DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
PARAPHRASE_PATTERN = args.paraphrase_pattern

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_pct_coordinates(choice_labels, DEBUG=False):
    """
    Calculate PCT coordinates from choice labels.
    choice_labels should be a list of 62 items, each either "unknown" or a string like "1", "2", "3", "4"
    Returns: (econ_result, soc_result, unknown_count)
    """
    econ_init = 0.38
    soc_init = 2.41

    econ_values = [
        [7, 5, 0, -2], #p1
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [7, 5, 0, -2], #p2
        [-7, -5, 0, 2],
        [6, 4, 0, -2],
        [7, 5, 0, -2],
        [-8, -6, 0, 2],
        [8, 6, 0, -2],
        [8, 6, 0, -1],
        [7, 5, 0, -3],
        [8, 6, 0, -1],
        [-7, -5, 0, 2],
        [-7, -5, 0, 1],
        [-6, -4, 0, 2],
        [6, 4, 0, -1],
        [0, 0, 0, 0],
        [0, 0, 0, 0], #p3
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-8, -6, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-10, -8, 0, 1],
        [-5, -4, 0, 1],
        [0, 0, 0, 0], #p4
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0], #p5
        [0, 0, 0, 0],
        [-9, -8, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0], #p6
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    soc_values = [
        [0, 0, 0, 0], #p1
        [-8, -6, 0, 2],
        [7, 5, 0, -2],
        [-7, -5, 0, 2],
        [-7, -5, 0, 2],
        [-6, -4, 0, 2],
        [7, 5, 0, -2],
        [0, 0, 0, 0], #p2
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-6, -4, 0, 2], #p3
        [7, 6, 0, -2],
        [-5, -4, 0, 2],
        [0, 0, 0, 0],
        [8, 4, 0, -2],
        [-7, -5, 0, 2],
        [-7, -5, 0, 3],
        [6, 4, 0, -3],
        [6, 3, 0, -2],
        [-7, -5, 0, 3],
        [-9, -7, 0, 2],
        [-8, -6, 0, 2],
        [7, 6, 0, -2],
        [-7, -5, 0, 2],
        [-6, -4, 0, 2],
        [-7, -4, 0, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [7, 5, 0, -3], #p4
        [-9, -6, 0, 2],
        [-8, -6, 0, 2],
        [-8, -6, 0, 2],
        [-6, -4, 0, 2],
        [-8, -6, 0, 2],
        [-7, -5, 0, 2],
        [-8, -6, 0, 2],
        [-5, -3, 0, 2],
        [-7, -5, 0, 2],
        [7, 5, 0, -2],
        [-6, -4, 0, 2],
        [-7, -5, 0, 2], #p5
        [-6, -4, 0, 2],
        [0, 0, 0, 0],
        [-7, -5, 0, 2],
        [-6, -4, 0, 2],
        [-7, -6, 0, 2], #p6
        [7, 6, 0, -2],
        [7, 5, 0, -2],
        [8, 6, 0, -2],
        [-8, -6, 0, 2],
        [-6, -4, 0, 2]
    ]
        
    econ_sum, soc_sum = 0, 0
    unknown_count = 0

    for i in range(min(62, len(choice_labels))): # 62 pct propositions
        if choice_labels[i] != "unknown": # skip unknowns
            choice_idx = int(choice_labels[i]) - 1  # convert to 0-indexed
            if 0 <= choice_idx < 4:  # validate choice index
                econ_sum += econ_values[i][choice_idx]
                soc_sum += soc_values[i][choice_idx]
            else:
                unknown_count += 1
        else:
            unknown_count += 1

    econ_result = econ_sum / 8.0
    soc_result = soc_sum / 19.5

    econ_result = econ_result + econ_init
    soc_result = soc_result + soc_init

    if DEBUG:
        print(f"Unknown responses: {unknown_count}")
        print(f"econ_result: {econ_result:.4f}, soc_result: {soc_result:.4f}")

    return econ_result, soc_result, unknown_count

# --- DATA GATHERING ---

def get_paraphrase_dirs():
    """Get all paraphrase directories matching the pattern"""
    if not os.path.exists(RESPONSES_DIR):
        return []
    
    from glob import glob
    pattern = os.path.join(RESPONSES_DIR, PARAPHRASE_PATTERN)
    dirs = [d for d in glob(pattern) if os.path.isdir(d)]
    return sorted(dirs)

def get_lang_from_filename(filename):
    basename = os.path.basename(filename)
    match = re.match(r'^([a-zA-Z]+)', basename)
    return match.group(1) if match else None

def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_choice_index(response_text, choices):
    """Extract choice index from response text"""
    # Try to find a numbered choice first (e.g., "1.", "2.")
    match = re.search(r'\b([1-4])', response_text)
    if match:
        return int(match.group(1)) - 1
        
    for i, choice_text in enumerate(choices):
        if choice_text.lower() in response_text.lower():
            return i

    return None

def process_intervention_responses(paraphrase_dir):
    """
    Process a single paraphrase directory and extract baseline vs intervention coordinates.
    Returns: (baseline_coords, intervention_coords, lang, unknown_counts)
    """
    paraphrase_name = os.path.basename(paraphrase_dir)
    print(f"\nProcessing paraphrase: {paraphrase_name}")
    
    # Get response files
    response_files = []
    for filename in os.listdir(paraphrase_dir):
        if filename.endswith('_responses.json'):
            response_files.append(os.path.join(paraphrase_dir, filename))
    
    if not response_files:
        print(f"  No response files found in {paraphrase_dir}")
        return None, None, None, {}
    
    # Process first response file (assuming single language)
    file_path = response_files[0]
    lang = get_lang_from_filename(file_path)
    if not lang:
        print(f"  Could not extract language from filename: {file_path}")
        return None, None, None, {}

    lang_upper = lang.upper()

    # Load choices data
    choices_path = os.path.join(DATA_DIR, f"{lang}.json")
    choices_data = load_json_file(choices_path)
    if not choices_data or 'choices' not in choices_data:
        print(f"  Skipping {lang}: Choices data not found or malformed at {choices_path}")
        return None, None, None, {}
        
    # Load responses data
    responses_data = load_json_file(file_path)
    if not responses_data:
        print(f"  Skipping {lang}: Responses data not found or malformed at {file_path}")
        return None, None, None, {}

    # Extract baseline and intervention choice labels
    baseline_labels = []
    intervention_labels = []
    
    for i, item in enumerate(responses_data):
        # Extract baseline response
        baseline_text = item.get('baseline_response', '')
        if not baseline_text:
            baseline_labels.append("unknown")
        else:
            baseline_idx = extract_choice_index(baseline_text, choices_data['choices'])
            if baseline_idx is None:
                baseline_labels.append("unknown")
            else:
                baseline_labels.append(str(baseline_idx + 1))  # Convert to 1-indexed
        
        # Extract intervention response
        intervention_text = item.get('intervention_response', '')
        if not intervention_text:
            intervention_labels.append("unknown")
        else:
            intervention_idx = extract_choice_index(intervention_text, choices_data['choices'])
            if intervention_idx is None:
                intervention_labels.append("unknown")
            else:
                intervention_labels.append(str(intervention_idx + 1))  # Convert to 1-indexed

    # Pad with "unknown" if we have fewer than 62 responses
    while len(baseline_labels) < 62:
        baseline_labels.append("unknown")
    while len(intervention_labels) < 62:
        intervention_labels.append("unknown")

    # Calculate PCT coordinates
    baseline_econ, baseline_soc, baseline_unknown = calculate_pct_coordinates(baseline_labels, DEBUG=False)
    intervention_econ, intervention_soc, intervention_unknown = calculate_pct_coordinates(intervention_labels, DEBUG=False)
    
    baseline_coords = (baseline_econ, baseline_soc)
    intervention_coords = (intervention_econ, intervention_soc)
    
    unknown_counts = {
        'baseline': baseline_unknown,
        'intervention': intervention_unknown
    }
    
    print(f"  {lang_upper} Baseline: Economic = {baseline_econ:.4f}, Social = {baseline_soc:.4f}, Unknown = {baseline_unknown}")
    print(f"  {lang_upper} Intervention: Economic = {intervention_econ:.4f}, Social = {intervention_soc:.4f}, Unknown = {intervention_unknown}")
    
    return baseline_coords, intervention_coords, lang_upper, unknown_counts

def gather_intervention_data():
    """
    Process all paraphrase directories and gather baseline vs intervention coordinates.
    Returns: (intervention_data, all_unknown_counts)
    """
    intervention_data = []
    all_unknown_counts = {}
    paraphrase_dirs = get_paraphrase_dirs()
    
    if not paraphrase_dirs:
        print("Error: No paraphrase directories found.")
        return [], {}
    
    print(f"Found {len(paraphrase_dirs)} paraphrase directories to process.")
    
    for para_dir in paraphrase_dirs:
        para_name = os.path.basename(para_dir)
        baseline_coords, intervention_coords, lang, unknown_counts = process_intervention_responses(para_dir)
        
        if baseline_coords and intervention_coords:
            intervention_data.append({
                'paraphrase': para_name,
                'language': lang,
                'baseline_economic': baseline_coords[0],
                'baseline_social': baseline_coords[1],
                'intervention_economic': intervention_coords[0],
                'intervention_social': intervention_coords[1],
                'economic_change': intervention_coords[0] - baseline_coords[0],
                'social_change': intervention_coords[1] - baseline_coords[1]
            })
            all_unknown_counts[para_name] = unknown_counts
    
    return intervention_data, all_unknown_counts

# --- PLOTTING FUNCTIONS ---

def plot_intervention_compass(intervention_data):
    """
    Plot baseline vs intervention results with arrows showing the change.
    """
    if not intervention_data:
        print("No intervention data available for plotting.")
        return

    print("\nGenerating Intervention Political Compass Plot...")

    plt.figure(figsize=(14, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Font and style settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'axes.linewidth': 1.5,
        'grid.linewidth': 1,
        'grid.color': 'gray'
    })

    # Extract data
    baseline_econ = [d['baseline_economic'] for d in intervention_data]
    baseline_soc = [d['baseline_social'] for d in intervention_data]
    intervention_econ = [d['intervention_economic'] for d in intervention_data]
    intervention_soc = [d['intervention_social'] for d in intervention_data]
    paraphrases = [d['paraphrase'] for d in intervention_data]

    # Plot baseline points
    plt.scatter(baseline_econ, baseline_soc, 
               c='blue', marker='o', s=100, alpha=0.7, 
               edgecolors='black', linewidth=1, 
               label='Baseline', zorder=5)

    # Plot intervention points
    plt.scatter(intervention_econ, intervention_soc, 
               c='red', marker='^', s=100, alpha=0.7, 
               edgecolors='black', linewidth=1, 
               label='Intervention', zorder=5)

    # Draw arrows showing the change
    for i in range(len(intervention_data)):
        plt.annotate('', 
                    xy=(intervention_econ[i], intervention_soc[i]), 
                    xytext=(baseline_econ[i], baseline_soc[i]),
                    arrowprops=dict(arrowstyle='->', 
                                  color='green', 
                                  lw=2, 
                                  alpha=0.8),
                    zorder=4)

    # Draw quadrants
    plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)

    # Add quadrant labels
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="black", linewidth=1)

    plt.text(0.95, 0.95, 'Authoritarian\nRight', transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=14, color='black', bbox=bbox_props, weight='bold')
    plt.text(0.05, 0.95, 'Authoritarian\nLeft', transform=plt.gca().transAxes,
             ha='left', va='top', fontsize=14, color='black', bbox=bbox_props, weight='bold')
    plt.text(0.95, 0.05, 'Libertarian\nRight', transform=plt.gca().transAxes,
             ha='right', va='bottom', fontsize=14, color='black', bbox=bbox_props, weight='bold')
    plt.text(0.05, 0.05, 'Libertarian\nLeft', transform=plt.gca().transAxes,
             ha='left', va='bottom', fontsize=14, color='black', bbox=bbox_props, weight='bold')

    # Labels and title
    plt.xlabel('Economic Axis (Left ← → Right)', fontsize=16, weight='bold', labelpad=10)
    plt.ylabel('Social Axis (Libertarian ← → Authoritarian)', fontsize=16, weight='bold', labelpad=10)
    # plt.title('Political Compass: Baseline vs Intervention Results', fontsize=18, weight='bold', pad=20)

    # Axis limits
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Grid
    plt.grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.6)
    plt.gca().minorticks_on()
    plt.gca().grid(which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.5)

    # Legend
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1, 0.75), fontsize=14, frameon=True, 
                       fancybox=True, shadow=False, facecolor='white', 
                       edgecolor='black')
    legend.get_frame().set_linewidth(1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "intervention_compass_with_arrows.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_intervention_statistics(intervention_data):
    """
    Generate and save intervention statistics.
    """
    if not intervention_data:
        return

    print("\nGenerating Intervention Statistics...")
    
    df = pd.DataFrame(intervention_data)
    
    # Calculate summary statistics
    stats = {
        'baseline_economic': {
            'mean': df['baseline_economic'].mean(),
            'std': df['baseline_economic'].std(),
            'min': df['baseline_economic'].min(),
            'max': df['baseline_economic'].max()
        },
        'baseline_social': {
            'mean': df['baseline_social'].mean(),
            'std': df['baseline_social'].std(),
            'min': df['baseline_social'].min(),
            'max': df['baseline_social'].max()
        },
        'intervention_economic': {
            'mean': df['intervention_economic'].mean(),
            'std': df['intervention_economic'].std(),
            'min': df['intervention_economic'].min(),
            'max': df['intervention_economic'].max()
        },
        'intervention_social': {
            'mean': df['intervention_social'].mean(),
            'std': df['intervention_social'].std(),
            'min': df['intervention_social'].min(),
            'max': df['intervention_social'].max()
        },
        'economic_change': {
            'mean': df['economic_change'].mean(),
            'std': df['economic_change'].std(),
            'min': df['economic_change'].min(),
            'max': df['economic_change'].max()
        },
        'social_change': {
            'mean': df['social_change'].mean(),
            'std': df['social_change'].std(),
            'min': df['social_change'].min(),
            'max': df['social_change'].max()
        }
    }
    
    # Calculate change magnitudes
    df['change_magnitude'] = np.sqrt(df['economic_change']**2 + df['social_change']**2)
    stats['change_magnitude'] = {
        'mean': df['change_magnitude'].mean(),
        'std': df['change_magnitude'].std(),
        'min': df['change_magnitude'].min(),
        'max': df['change_magnitude'].max()
    }
    
    # Save detailed data
    df.to_csv(os.path.join(OUTPUT_DIR, "intervention_detailed_data.csv"), index=False)
    
    # Save statistics
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "intervention_statistics.csv"))
    
    # Print summary
    print(f"\nIntervention Effect Summary:")
    print(f"Average Economic Change: {df['economic_change'].mean():.4f} ± {df['economic_change'].std():.4f}")
    print(f"Average Social Change: {df['social_change'].mean():.4f} ± {df['social_change'].std():.4f}")
    print(f"Average Change Magnitude: {df['change_magnitude'].mean():.4f} ± {df['change_magnitude'].std():.4f}")
    
    # Test statistical significance (t-test)
    from scipy import stats as scipy_stats
    
    # Test if changes are significantly different from zero
    econ_ttest = scipy_stats.ttest_1samp(df['economic_change'], 0)
    soc_ttest = scipy_stats.ttest_1samp(df['social_change'], 0)
    
    print(f"\nStatistical Significance Tests (vs. no change):")
    print(f"Economic Change: t={econ_ttest.statistic:.4f}, p={econ_ttest.pvalue:.4f}")
    print(f"Social Change: t={soc_ttest.statistic:.4f}, p={soc_ttest.pvalue:.4f}")
    
    return stats

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("="*60)
    print("POLITICAL COMPASS INTERVENTION ANALYSIS")
    print("="*60)
    
    # Gather intervention data
    intervention_data, all_unknown_counts = gather_intervention_data()
    
    if not intervention_data:
        print("No intervention data found. Exiting.")
        exit(1)
    
    # Save unknown counts
    unk_file = os.path.join(OUTPUT_DIR, "unknown_counts.json")
    with open(unk_file, 'w') as f:
        json.dump(all_unknown_counts, f, indent=2)
    print(f"Unknown counts saved to: {unk_file}")
    
    # Generate plots
    plot_intervention_compass(intervention_data)
    
    # Generate statistics
    stats = generate_intervention_statistics(intervention_data)
    
    # Save all data
    data_file = os.path.join(OUTPUT_DIR, "intervention_data.json")
    with open(data_file, 'w') as f:
        json.dump(intervention_data, f, indent=2)
    
    stats_file = os.path.join(OUTPUT_DIR, "intervention_summary_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
