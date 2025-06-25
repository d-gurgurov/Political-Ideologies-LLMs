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
parser = argparse.ArgumentParser(description="Analyze political compass responses from batch processing.")
parser.add_argument('--responses_dir', type=str, default='outputs',
                    help='Directory containing paraphrase subdirectories with response JSON files')
parser.add_argument('--data_dir', type=str, default='data',
                    help='Directory containing choice files per language')
parser.add_argument('--output_dir', type=str, default='batch_analysis',
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

def process_paraphrase_dir(paraphrase_dir):
    """Process a single paraphrase directory and return coordinates and unknown counts for all languages"""
    paraphrase_name = os.path.basename(paraphrase_dir)
    print(f"\nProcessing paraphrase: {paraphrase_name}")
    
    coordinates = {}
    unknown_counts = {}
    
    # Get all response files in this paraphrase directory
    response_files = []
    for filename in os.listdir(paraphrase_dir):
        if filename.endswith('_responses.json'):
            response_files.append(os.path.join(paraphrase_dir, filename))
    
    if not response_files:
        print(f"  No response files found in {paraphrase_dir}")
        return coordinates, unknown_counts
    
    for file_path in response_files:
        lang = get_lang_from_filename(file_path)
        if not lang:
            print(f"  Could not extract language from filename: {file_path}")
            continue

        lang_upper = lang.upper()

        # Load choices data
        choices_path = os.path.join(DATA_DIR, f"{lang}.json")
        choices_data = load_json_file(choices_path)
        if not choices_data or 'choices' not in choices_data:
            print(f"  Skipping {lang}: Choices data not found or malformed at {choices_path}")
            continue
            
        # Load responses data
        responses_data = load_json_file(file_path)
        if not responses_data:
            print(f"  Skipping {lang}: Responses data not found or malformed at {file_path}")
            continue

        # Extract choice labels for all 62 questions
        choice_labels = []
        for i, item in enumerate(responses_data):
            response_text = item.get('response', '') # NORMALLY IT SHOULD BE response
            if not response_text:
                choice_labels.append("unknown")
                continue

            choice_idx = extract_choice_index(response_text, choices_data['choices'])
            if choice_idx is None:
                choice_labels.append("unknown")
            else:
                choice_labels.append(str(choice_idx + 1))  # Convert back to 1-indexed for PCT function

        # Pad with "unknown" if we have fewer than 62 responses
        while len(choice_labels) < 62:
            choice_labels.append("unknown")

        # Calculate PCT coordinates and get unknown count
        econ_coord, soc_coord, unknown_count = calculate_pct_coordinates(choice_labels, DEBUG=False)
        coordinates[lang_upper] = (econ_coord, soc_coord)
        unknown_counts[lang_upper] = unknown_count
        
        print(f"  {lang_upper}: Economic = {econ_coord:.4f}, Social = {soc_coord:.4f}, Unknown = {unknown_count}")

    return coordinates, unknown_counts

def gather_all_coordinates():
    """
    Process all paraphrase directories and gather coordinates and unknown counts.
    Returns: (all_coordinates, all_unknown_counts)
    all_coordinates: {paraphrase: {language: (econ, soc)}}
    all_unknown_counts: {paraphrase: {language: unknown_count}}
    """
    all_coordinates = {}
    all_unknown_counts = {}
    paraphrase_dirs = get_paraphrase_dirs()
    
    if not paraphrase_dirs:
        print("Error: No paraphrase directories found.")
        return {}, {}
    
    print(f"Found {len(paraphrase_dirs)} paraphrase directories to process.")
    
    for para_dir in paraphrase_dirs:
        para_name = os.path.basename(para_dir)
        coordinates, unknown_counts = process_paraphrase_dir(para_dir)
        if coordinates:
            all_coordinates[para_name] = coordinates
            all_unknown_counts[para_name] = unknown_counts
    
    return all_coordinates, all_unknown_counts

def save_unknown_counts_to_json(all_unknown_counts, output_file="unknown_counts.json"):
    """Save unknown counts to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_unknown_counts, f, indent=2, ensure_ascii=False)
        print(f"\nUnknown counts saved to {output_file}")
    except Exception as e:
        print(f"Error saving unknown counts to {output_file}: {e}")

# --- ANALYSIS FUNCTIONS ---

def calculate_averages_and_std(all_coordinates):
    """
    Calculate averages and standard deviations across paraphrases for each language.
    Returns: (avg_coordinates, std_coordinates, detailed_data)
    """
    # Collect all data points
    language_data = defaultdict(lambda: {'econ': [], 'soc': []})
    detailed_data = []
    
    for paraphrase, lang_coords in all_coordinates.items():
        for language, (econ, soc) in lang_coords.items():
            language_data[language]['econ'].append(econ)
            language_data[language]['soc'].append(soc)
            detailed_data.append({
                'paraphrase': paraphrase,
                'language': language,
                'economic': econ,
                'social': soc
            })
    
    # Calculate averages and standard deviations
    avg_coordinates = {}
    std_coordinates = {}
    
    for language in language_data:
        econ_values = language_data[language]['econ']
        soc_values = language_data[language]['soc']
        
        avg_coordinates[language] = (np.mean(econ_values), np.mean(soc_values))
        std_coordinates[language] = (np.std(econ_values), np.std(soc_values))
    
    return avg_coordinates, std_coordinates, detailed_data

# --- PLOTTING FUNCTIONS ---

def plot_averaged_compass_with_error_bars(avg_coordinates, std_coordinates):
    """
    Print-friendly version of the averaged political compass plot with strong visual contrast
    and enhanced text/grid visibility for publication use.
    """
    if not avg_coordinates:
        print("No coordinates available to plot averaged compass.")
        return

    print("\nGenerating Print-Optimized Averaged PCT Political Compass with Error Bars...")

    plt.figure(figsize=(12, 9))  # ACL-friendly size
    plt.style.use('seaborn-v0_8-whitegrid')

    # Font and line settings for better print visibility
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 10,
        'axes.linewidth': 2,
        'grid.linewidth': 1.5,
        'grid.color': 'gray'
    })

    # Extract coordinates
    languages = sorted(list(avg_coordinates.keys()))
    econ_coords = [avg_coordinates[lang][0] for lang in languages]
    soc_coords = [avg_coordinates[lang][1] for lang in languages]
    econ_stds = [std_coordinates[lang][0] for lang in languages]
    soc_stds = [std_coordinates[lang][1] for lang in languages]

    # Define styles
    distinct_colors = [
        '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
        '#1ABC9C', '#E67E22', '#34495E', '#F1C40F', '#E91E63',
        '#00BCD4', '#4CAF50', '#FF5722', '#607D8B', '#795548',
        '#8BC34A', '#FF9800', '#673AB7', '#009688', '#CDDC39'
    ]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h',
               'H', '+', 'x', 'P', 'X', '8', '1', '2', '3', '4']
    line_styles = ['-', '--', '-.', ':'] * 5

    # Extend styles if needed
    while len(distinct_colors) < len(languages):
        distinct_colors *= 2
    while len(markers) < len(languages):
        markers *= 2
    while len(line_styles) < len(languages):
        line_styles *= 2

    language_styles = {
        lang: {
            'color': distinct_colors[i],
            'marker': markers[i],
            'linestyle': line_styles[i],
            'size': 14 + (i % 3) * 2
        }
        for i, lang in enumerate(languages)
    }

    # Plot
    for i, lang in enumerate(languages):
        style = language_styles[lang]
        plt.errorbar(
            econ_coords[i], soc_coords[i],
            xerr=econ_stds[i], yerr=soc_stds[i],
            fmt=style['marker'],
            markersize=style['size'],
            capsize=7, capthick=2,
            color=style['color'],
            markeredgecolor='black',
            markeredgewidth=1,
            elinewidth=2,
            linestyle=style['linestyle'],
            alpha=0.85,
            label=lang,
            zorder=5
        )

    # Draw quadrants
    plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.9)
    plt.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.9)

    # Add quadrant labels
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95, edgecolor="black", linewidth=1.5)

    plt.text(0.95, 0.95, 'Authoritarian\nRight', transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=16, color='black', bbox=bbox_props, weight='bold')
    plt.text(0.05, 0.95, 'Authoritarian\nLeft', transform=plt.gca().transAxes,
             ha='left', va='top', fontsize=16, color='black', bbox=bbox_props, weight='bold')
    plt.text(0.95, 0.05, 'Libertarian\nRight', transform=plt.gca().transAxes,
             ha='right', va='bottom', fontsize=16, color='black', bbox=bbox_props, weight='bold')
    plt.text(0.05, 0.05, 'Libertarian\nLeft', transform=plt.gca().transAxes,
             ha='left', va='bottom', fontsize=16, color='black', bbox=bbox_props, weight='bold')

    # Labels
    plt.xlabel('Economic Axis (Left ← → Right)', fontsize=20, weight='bold', labelpad=12)
    plt.ylabel('Social Axis (Libertarian ← → Authoritarian)', fontsize=20, weight='bold', labelpad=12)

    # Axis limits
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Grid
    plt.grid(True, linestyle='--', linewidth=1.5, color='gray', alpha=0.6)
    plt.gca().minorticks_on()
    plt.gca().grid(which='minor', linestyle=':', linewidth=1.0, color='lightgray', alpha=0.5)

    # Legend
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16,
                        frameon=True, fancybox=True, shadow=False,
                        facecolor='white', edgecolor='black', ncol=1)
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "enhanced_averaged_compass_with_error_bars.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def plot_all_paraphrases_scatter(all_coordinates):
    """
    Generate a high-contrast scatter plot showing all paraphrases and languages
    with confidence ellipses and quadrant annotations.
    Optimized for print and academic publication.
    """
    if not all_coordinates:
        print("No coordinates available to plot all paraphrases.")
        return

    print("\nGenerating High-Contrast All Paraphrases Scatter Plot...")

    fig = plt.figure(figsize=(14, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    ax_main = plt.subplot(1, 1, 1)

    # Gather languages
    all_languages = sorted({lang for coords in all_coordinates.values() for lang in coords})

    # Define visual styles
    distinct_colors = [
        '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
        '#1ABC9C', '#E67E22', '#34495E', '#F1C40F', '#E91E63',
        '#00BCD4', '#4CAF50', '#FF5722', '#607D8B', '#795548',
        '#8BC34A', '#FF9800', '#673AB7', '#009688', '#CDDC39'
    ]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h',
               'H', '+', 'x', 'P', 'X', '8', '1', '2', '3', '4']

    while len(distinct_colors) < len(all_languages):
        distinct_colors *= 2
    while len(markers) < len(all_languages):
        markers *= 2

    language_styles = {
        lang: {'color': distinct_colors[i], 'marker': markers[i]}
        for i, lang in enumerate(all_languages)
    }

    for lang in all_languages:
        econ_coords, soc_coords = [], []
        for paraphrase, lang_coords in all_coordinates.items():
            if lang in lang_coords:
                econ, soc = lang_coords[lang]
                econ_coords.append(econ)
                soc_coords.append(soc)

        if not econ_coords:
            continue

        ax_main.scatter(
            econ_coords, soc_coords,
            c=language_styles[lang]['color'],
            marker=language_styles[lang]['marker'],
            label=lang,
            s=120, alpha=0.85,
            edgecolors='black', linewidth=1.2,
            zorder=6
        )

        if len(econ_coords) > 2:
            from matplotlib.patches import Ellipse
            from scipy.stats import chi2

            coords = np.column_stack((econ_coords, soc_coords))
            mean = np.mean(coords, axis=0)
            cov = np.cov(coords.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
            angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvals * chi2.ppf(0.95, df=2))

            ellipse = Ellipse(
                xy=mean, width=width, height=height, angle=angle,
                facecolor=language_styles[lang]['color'],
                edgecolor=language_styles[lang]['color'],
                linestyle='--', linewidth=2.5, alpha=0.18, zorder=1
            )
            ax_main.add_patch(ellipse)

    # Quadrants and labels
    plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.9)
    plt.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.9)

    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.9, linewidth=1.5)

    plt.text(0.95, 0.95, 'Authoritarian\nRight', transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=16, bbox=bbox_props, weight='bold')
    plt.text(0.05, 0.95, 'Authoritarian\nLeft', transform=plt.gca().transAxes,
             ha='left', va='top', fontsize=16, bbox=bbox_props, weight='bold')
    plt.text(0.95, 0.05, 'Libertarian\nRight', transform=plt.gca().transAxes,
             ha='right', va='bottom', fontsize=16, bbox=bbox_props, weight='bold')
    plt.text(0.05, 0.05, 'Libertarian\nLeft', transform=plt.gca().transAxes,
             ha='left', va='bottom', fontsize=16, bbox=bbox_props, weight='bold')

    # Labels and limits
    plt.xlabel('Economic Axis (Left ← → Right)', fontsize=20, weight='bold', labelpad=10)
    plt.ylabel('Social Axis (Libertarian ← → Authoritarian)', fontsize=20, weight='bold', labelpad=10)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Grid and ticks
    plt.grid(True, linestyle='--', linewidth=1.2, color='gray', alpha=0.7)
    plt.gca().minorticks_on()
    plt.gca().grid(which='minor', linestyle=':', linewidth=0.8, alpha=0.5)

    # Legend
    legend = plt.legend(
        bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14,
        frameon=True, fancybox=True, facecolor='white', edgecolor='black'
    )
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "all_paraphrases_scatter_grouped.png"), dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(avg_coordinates, std_coordinates, detailed_data):
    """
    Generate and save summary statistics
    """
    print("\nGenerating Summary Statistics...")
    
    df = pd.DataFrame(detailed_data)
    
    # Calculate statistics by language
    lang_stats = df.groupby('language').agg({
        'economic': ['mean', 'std', 'min', 'max'],
        'social': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # Calculate statistics by paraphrase
    para_stats = df.groupby('paraphrase').agg({
        'economic': ['mean', 'std', 'min', 'max'],
        'social': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # Save to files
    lang_stats_file = os.path.join(OUTPUT_DIR, "language_statistics.csv")
    para_stats_file = os.path.join(OUTPUT_DIR, "paraphrase_statistics.csv")
    
    lang_stats.to_csv(lang_stats_file)
    para_stats.to_csv(para_stats_file)
    
    print(f"Language statistics saved to: {lang_stats_file}")
    print(f"Paraphrase statistics saved to: {para_stats_file}")
    
    # Print summary
    print(f"\nOverall Statistics:")
    print(f"Economic axis - Mean: {df['economic'].mean():.4f}, Std: {df['economic'].std():.4f}")
    print(f"Social axis - Mean: {df['social'].mean():.4f}, Std: {df['social'].std():.4f}")
    print(f"Total data points: {len(df)}")
    print(f"Languages: {df['language'].nunique()}")
    print(f"Paraphrases: {df['paraphrase'].nunique()}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("="*60)
    print("BATCH POLITICAL COMPASS ANALYSIS WITH PCT COORDINATES")
    print("="*60)
    
    # Gather all coordinates
    all_coordinates, all_unknown_counts = gather_all_coordinates()
    
    # Save unknown counts to JSON file
    unk_file = os.path.join(OUTPUT_DIR, "unknown_counts.json")
    save_unknown_counts_to_json(all_unknown_counts, output_file=unk_file)
    
    if not all_coordinates:
        print("No data found. Exiting.")
        exit(1)
    
    # Calculate averages and standard deviations
    avg_coordinates, std_coordinates, detailed_data = calculate_averages_and_std(all_coordinates)
    
    # Generate plots
    plot_averaged_compass_with_error_bars(avg_coordinates, std_coordinates)
    
    # Generate summary statistics
    generate_summary_statistics(avg_coordinates, std_coordinates, detailed_data)
    
    # Save all data to files
    all_coords_file = os.path.join(OUTPUT_DIR, "all_coordinates.json")
    avg_coords_file = os.path.join(OUTPUT_DIR, "averaged_coordinates.json")
    detailed_data_file = os.path.join(OUTPUT_DIR, "detailed_data.json")
    
    with open(all_coords_file, 'w') as f:
        json.dump(all_coordinates, f, indent=2)
    
    with open(avg_coords_file, 'w') as f:
        # Convert tuples to lists for JSON serialization
        avg_json = {lang: list(coords) for lang, coords in avg_coordinates.items()}
        std_json = {lang: list(coords) for lang, coords in std_coordinates.items()}
        json.dump({"averages": avg_json, "standard_deviations": std_json}, f, indent=2)
    
    with open(detailed_data_file, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    
    print(f"\nAll data saved to:")
    print(f"  - All coordinates: {all_coords_file}")
    print(f"  - Averaged coordinates: {avg_coords_file}")
    print(f"  - Detailed data: {detailed_data_file}")
    
    print("\n" + "="*60)
    print("BATCH ANALYSIS COMPLETE!")
    print("="*60)