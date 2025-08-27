import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def load_density(score_mathilde, densities):
    tilsscore_calibration = {}

    for line in score_mathilde.itertuples(index=False):
        tnumber = line[0]
        tilsscore = line[1]
        for key in densities:
            if str(tnumber) in str(key):
                tilsscore_calibration[key] = (tilsscore, densities[key]) # {tnumber: (pathologist score, density)}

    return tilsscore_calibration


def load_tilscount(tilsscore, tilscounts):
    score_counts = {}

    for line in tilsscore.itertuples(index=False, name=None):
        tnumber = line[0]
        tilsscore = line[1]
        for key in tilscounts:
            if str(tnumber) in str(key):
                score_counts[key] = (tilsscore, tilscounts[key]) # {tnumber: (pathologist score, TILs count)}
    
    return score_counts
    

def clustered_histogram(scoredict):
    grouped = defaultdict(list)
    for key, (x, y) in scoredict.items():
        grouped[x].append(y)

    sorted_groups = sorted(grouped.items())

    bar_width = 0.8
    inter_bar_spacing = 0.1
    cluster_spacing = 1.5
    fig, ax = plt.subplots(figsize=(12, 6))
    x_ticks = []
    x_labels = []
    x_cursor = 0 
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_groups)))

    medians = {}
        
    for i, (x_val, y_vals) in enumerate(sorted_groups):
        group_color = colors[i]
        n = len(y_vals)

        for j, y in enumerate(y_vals):
            bar_x = x_cursor + j * (bar_width + inter_bar_spacing)
            ax.bar(bar_x, y, width=bar_width, color=group_color)

        group_width = (n - 1) * (bar_width + inter_bar_spacing)
        group_center = x_cursor + group_width / 2
        x_ticks.append(group_center)
        x_labels.append(str(x_val))

        # Median line 
        median_y = np.median(y_vals)
        medians[x_val] = median_y
        first_bar_x = x_cursor - bar_width / 2
        last_bar_x = x_cursor + (n - 1) * (bar_width + inter_bar_spacing) + bar_width / 2

        ax.hlines(y=median_y, xmin=first_bar_x, xmax=last_bar_x,
                colors='black', linewidth=0.5)

        x_cursor += n * (bar_width + inter_bar_spacing) + cluster_spacing

    # Plot
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=300)
    ax.set_xlabel('Pathologist B TILs score')
    ax.set_ylabel('Lymphocyte count per slide')
    ax.set_title('Lymphocyte count clustered per TILs score')
    plt.tight_layout()
    plt.savefig("histograms/Clustered_counts_pathb_histogram.png", dpi=300)

    return medians


def plot_median_lymphocyte_per_tilscore(scoredict):
    grouped = defaultdict(list)
    for key, (til_score, lymph_count) in scoredict.items():
        if til_score is not None and lymph_count is not None:
            grouped[til_score].append(lymph_count)

    tilsscore = []
    medians = []

    for tils, counts in grouped.items():
        tilsscore.append(tils)
        medians.append(np.median(counts))

    # Sort by TILs score
    tilsscore, medians = zip(*sorted(zip(tilsscore, medians)))

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tilsscore, y=medians, palette="Blues_d")
    plt.xlabel("Pathologist TILs Score")
    plt.ylabel("Median Lymphocyte Count")
    plt.title("Median Lymphocyte Count per TILs Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("histograms/tilsscore_median_lymphocyte_count.png", dpi=300)


def main():
    # Location of TILs scores by Mathilde, use column H ('avgTIL), average of two pathologists
    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/calibration/overlap_Mathilde.tsv", "r") as f:
        avg_mathilde = pd.read_csv(f, sep="\t")

    # TILs scores per pathologist are B and E
    patha_mathilde = pd.read_excel("/home/t.leppers/tils/TILsscore_Mathilde.xlsx", sheet_name="Sheet1", usecols=["t_number", "TIL_score_m"])
    pathb_mathilde = pd.read_excel("/home/t.leppers/tils/TILsscore_Mathilde.xlsx", sheet_name="Sheet1", usecols=["t_number", "TIL_score_j"])

    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/8409_tilscounts.pkl", "rb") as f:
        tilscounts = pickle.load(f)
    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/8409_avg_densities.pkl", "rb") as f:
        avg_densities = pickle.load(f)
    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/8409_max_densities.pkl", "rb") as f:
        max_densities = pickle.load(f)

    # Pathologist-specific clustering
    patha_avg_density = load_density(patha_mathilde, avg_densities)
    patha_max_density = load_density(patha_mathilde, max_densities)
    patha_counts = load_tilscount(patha_mathilde, tilscounts)
    clustered_histogram(patha_avg_density)
    clustered_histogram(patha_max_density)
    clustered_histogram(patha_counts)
    pathb_avg_density = load_density(pathb_mathilde, avg_densities)
    pathb_max_density = load_density(pathb_mathilde, max_densities)
    pathb_counts = load_tilscount(pathb_mathilde, tilscounts)
    clustered_histogram(pathb_avg_density)
    clustered_histogram(pathb_max_density)
    clustered_histogram(pathb_counts)

    # Clustering over average TILsscore
    tilscounts = load_tilscount(avg_mathilde, tilscounts)
    avg_densities = load_density(avg_mathilde, avg_densities)
    max_densities = load_density(avg_mathilde, max_densities)
    clustered_histogram(avg_densities)
    clustered_histogram(max_densities)
    clustered_histogram(tilscounts)

    # Median plots
    plot_median_lymphocyte_per_tilscore(avg_densities)
    plot_median_lymphocyte_per_tilscore(max_densities)
    plot_median_lymphocyte_per_tilscore(tilscounts)


if __name__ == "__main__":
    main()
