import json
import pandas as pd
import numpy as np
import pickle
import sys
import os
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, auc
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
plt.rcParams.update({'font.size': 6})
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr, fisher_exact
import scikit_posthocs as sp
import itertools
from statannotations.Annotator import Annotator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TLS/validation')))
from kaplan_meier import retrieve_mpp, calculate_area
from dython.nominal import associations
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def load_data(clinical_data_path):
    # Retrieve survival data
    clinical_data = pd.read_excel(clinical_data_path, engine='openpyxl')
    clinical_data.drop(columns=['specimen_coll_group_id', 'tissue_number_blockid',
       'year_diagnose', 'surgery_final', 'vital_status', 'er_percentage', 'her2_score', 'p16_percentage',
       'p16_score', 'ki67_percentage', 'ki67_category'], inplace=True) # Remove features that are not needed
    
    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/8409_average_densities.pkl", "rb") as f:
        avg_densities = pickle.load(f)
    avg_densities = {str(key).split(' ')[0]: value for key, value in avg_densities.items()}

    # Add average lymphocyte density to clinical data
    clinical_data['avg_lymphocyte_density'] = clinical_data['tissue_number'].apply(
        lambda t: avg_densities.get(str(t), np.nan)
    )

    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/8409_tilscounts.pkl", "rb") as f:
        tilscounts = pickle.load(f)
    tilscounts = {str(key).split(' ')[0]: value for key, value in tilscounts.items()}
    

    # Add lymphocyte data
    clinical_data['lymphocyte_count'] = clinical_data['tissue_number'].apply(
        lambda t: tilscounts.get(str(t), np.nan)
    )

    path_tilsscore = pd.read_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/calibration/overlap_Mathilde.tsv", sep='\t')
    path_tilsscore_dict = dict(zip(path_tilsscore.iloc[:, 0].astype(str), path_tilsscore.iloc[:, 1]))
    clinical_data['pathologist_tilsscore'] = clinical_data['tissue_number'].apply(
        lambda t: path_tilsscore_dict.get(str(t), np.nan)
    )

    density_variances = pd.read_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/density_variances.csv") # Variance
    variance_map = {str(row['image_name']).split(" ")[0]: row['variance'] for _, row in density_variances.iterrows()}
    clinical_data['lymph_density_var'] = clinical_data['tissue_number'].apply(lambda t: variance_map.get(str(t), np.nan))

    density_entropy = pd.read_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/density_entropy.csv") # Entropy
    entropy_map = {str(row['image_name']).split(" ")[0]: row['entropy'] for _, row in density_entropy.iterrows()}
    clinical_data['lymph_density_entropy'] = clinical_data['tissue_number'].apply(lambda t: entropy_map.get(str(t), np.nan))

    density_moran_raw = pd.read_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/density_moran_raw.csv") # Moran's I raw
    moran_raw_map = {str(row['image_name']).split(" ")[0]: row['moran_I'] for _, row in density_moran_raw.iterrows()}
    clinical_data['lymph_density_moran_raw'] = clinical_data['tissue_number'].apply(lambda t: moran_raw_map.get(str(t), np.nan))

    density_moran_hotspot = pd.read_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/metadata/density/density_moran_hotspots.csv") # Moran's I hotspot
    moran_hotspot_map = {str(row['image_name']).split(" ")[0]: row['hotspot'] for _, row in density_moran_hotspot.iterrows()}
    clinical_data['lymph_density_moran_hotspot'] = clinical_data['tissue_number'].apply(lambda t: moran_hotspot_map.get(str(t), np.nan))


    # Add duct data
    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/ducts/duct_count.pkl", "rb") as f:
        duct_count = pickle.load(f)
    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/ducts/duct_perimeter_avg.pkl", "rb") as f:
        duct_perimeter_avg = pickle.load(f)
    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/ducts/duct_size_avg.pkl", "rb") as f:
        duct_size_avg = pickle.load(f)
    with open("/home/t.leppers/dev/pathology-projects/EllogonTILs/ducts/duct_size_total.pkl", "rb") as f:
        duct_size_total = pickle.load(f)
    ducts_poly_area = pd.read_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/analysis/plots/ducts/duct_associated_area.csv")

    clinical_data['duct_count'] = clinical_data['tissue_number'].apply(
        lambda t: duct_count.get(str(t), np.nan)
    )
    clinical_data['duct_perimeter_avg'] = clinical_data['tissue_number'].apply(
        lambda t: duct_perimeter_avg.get(str(t), np.nan)
    )
    clinical_data['duct_size_avg'] = clinical_data['tissue_number'].apply(
        lambda t: duct_size_avg.get(str(t), np.nan)
    )
    clinical_data['duct_size_total'] = clinical_data['tissue_number'].apply(
        lambda t: duct_size_total.get(str(t), np.nan)
    )
    duct_area_map = {
        str(row['tnumber']).split('_')[0]: row['duct_associated_area_mm2']
        for _, row in ducts_poly_area.iterrows()
    }
    clinical_data['ducts_poly_area_mm2'] = clinical_data['tissue_number'].apply(
        lambda t: duct_area_map.get(str(t), np.nan)
    )
    clinical_data['lymphocyte_area_mm2'] = clinical_data['lymphocyte_count'] * np.pi * (0.004 ** 2)  # Given average lymphocyte diameter of 8 micrometer
    clinical_data['duct_associated_area_mm2'] = clinical_data['lymphocyte_area_mm2'] / clinical_data['ducts_poly_area_mm2'].replace(0, np.nan) 


    # Add TLS data
    with open("/home/t.leppers/dev/pathology-projects/TLS/mappings/ppid_mapping.json", "r") as f:
        ppid_mapping = json.load(f)

    annotated_slides_dir = "/projects/dcis-recurrence-tjarda/Aperio_container_outputs/images/converted"
    annotated_slides = [f for f in os.listdir(annotated_slides_dir) if f.endswith(".geojson") and os.path.isfile(os.path.join(annotated_slides_dir, f))]

    TLS = []
    TLS_area = []
    TLS_count = []
    # Calculate TLS area within loop
    for ppid in clinical_data['ppid']:
        if ppid in ppid_mapping:
            slidename = ppid_mapping[ppid].replace(".svs", ".geojson")
            if slidename in annotated_slides:
                area_p, count = calculate_area(os.path.join(annotated_slides_dir, slidename))
                area = area_p * retrieve_mpp(slidename.replace(".geojson", ".svs"))
                total_area = area.sum()
                TLS_area.append(total_area)
                TLS_count.append(count)
            else:
                TLS_area.append(np.nan)
                TLS_count.append(np.nan)
        else:
            TLS_area.append(np.nan)
            TLS_count.append(np.nan)

    avg_area = np.nansum(TLS_area) / len(TLS_area) # Ignore NaN values
    for area in TLS_area:
        if area >= avg_area:
            TLS.append('High')
        else:
            TLS.append('Low')

    clinical_data['TLS'] = TLS
    clinical_data['TLS_count'] = TLS_count
    clinical_data['TLS_area'] = TLS_area

    clinical_data.to_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/analysis/clinical_data_processed.csv", index=False)


def binarize_data(clinical_data):
    bool_data = clinical_data.drop(columns=['ppid', 'tissue_number']).copy()
    bool_data['first_subseq_event'] = bool_data['first_subseq_event'].fillna('no event')
    for col in ['er', 'pr', 'her2', 'p16_conclusion']:
        bool_data[col] = bool_data[col].astype(str).str.lower().replace({'negative': 0, 'positive': 1})
        bool_data[col] = bool_data[col].where(bool_data[col].isin([0, 1]), np.nan)
    bool_data['p53_percentage'] = bool_data['p53_percentage'].fillna(0).apply(lambda x: 1 if x > 10 else 0) # WHO criteria for p53 positivity
    # Binarize COX2 score (low=1, high=2,3)
    bool_data['cox2_score'] = bool_data['cox2_score'].apply(lambda x: {1: 0, 2: 1, 3: 1}.get(x, np.nan)).astype('Int64')
    # Binarize grade (low=1,2, high=3)
    bool_data['grade'] = bool_data['grade'].apply(lambda x: {1: 0, 2: 0, 3: 1}.get(x, np.nan)).astype('Int64')
    bool_data['TLS'] = bool_data['TLS'].apply(lambda x: 1 if x == 'High' else (0 if x == 'Low' else np.nan))
    bool_data['first_subseq_event'] = bool_data['first_subseq_event'].replace({'ipsilateral ibc': 1, 'no event': 0})
    bool_data.loc[(bool_data['first_subseq_event'] == 1) & (bool_data['event_months'] > 60), 'first_subseq_event'] = 0 # Only consider events within 5 years
    bool_data.loc[~bool_data['first_subseq_event'].isin([0, 1]), 'first_subseq_event'] = np.nan

    bool_data.to_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/analysis/processed_bool_data.csv", index=False)


def clinicaltableoverview(clinical_data):
    # Filter the df to include only the relevant values
    clinical_data['first_subseq_event'] = clinical_data['first_subseq_event'].fillna('NA')
    filtered_data = clinical_data[clinical_data['first_subseq_event'].isin(['NA', 'ipsilateral ibc'])].copy()
    filtered_data['age_group'] = filtered_data['age_diagnose'].apply(lambda x: '<50' if x < 50 else '≥50')

    # Group and summarize the data
    summary = filtered_data.groupby('first_subseq_event').agg(
        age_mean_range=('age_diagnose', lambda x: f"{x.mean():.1f} ({x.min()}–{x.max()})"),
        age_lt_50=('age_group', lambda x: (x == '<50').sum()),
        age_ge_50=('age_group', lambda x: (x == '≥50').sum()),
        rt_0=('radiotherapy', lambda x: (x == 0).sum()),
        rt_1=('radiotherapy', lambda x: (x == 1).sum()),
        grade_1=('grade', lambda x: (x == 1).sum()),
        grade_2=('grade', lambda x: (x == 2).sum()),
        grade_3=('grade', lambda x: (x == 3).sum()),
        er_pos=('er', lambda x: (x == 'positive').sum()),
        er_neg=('er', lambda x: (x == 'negative').sum()),
        er_unknown=('er', lambda x: (x == 'unknown').sum()),
        pr_pos=('pr', lambda x: (x == 'positive').sum()),
        pr_neg=('pr', lambda x: (x == 'negative').sum()),
        pr_unknown=('pr', lambda x: (x == 'unknown').sum()),
        her2_pos=('her2', lambda x: (x == 'positive').sum()),
        her2_neg=('her2', lambda x: (x == 'negative').sum()),
        her2_unknown=('her2', lambda x: (x == 'unknown').sum()),
        p16_pos=('p16_conclusion', lambda x: (x == 'Positive').sum()),
        p16_neg=('p16_conclusion', lambda x: (x == 'Negative').sum()),
        p16_unknown=('p16_conclusion', lambda x: (x == 'nan').sum()),
        p_53_pos=('p53_percentage', lambda x: (x > 10).sum()),  # WHO criteria for p53 positivity
        p_53_neg=('p53_percentage', lambda x: (x <= 10).sum()),
        p_53_unknown=('p53_percentage', lambda x: (x.isna()).sum()),
        cox2_high=('cox2_score', lambda x: (x.isin([2, 3])).sum()),
        cox2_low=('cox2_score', lambda x: (x == 1).sum()),
        cox2_unknown=('cox2_score', lambda x: (x.isna()).sum()),
    )

    summary_t = summary.transpose()
    summary_t.to_excel("plots/clinical_table.xlsx")


def correlation_matrix(bool_data):
    columns = [column for column in clinical_data.columns if column != 'event_months']

    results = associations(clinical_data[columns], nominal_columns='auto', plot=False)
    corr_matrix = results['corr']
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    plt.title('Correlation Matrix of Clinical Data')
    plt.xticks(rotation=45, ha='right')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig("plots/multivariate/correlation_matrix.png", dpi=300)


def duct_lymphocyte_correlation(bool_data):
    lymphocyte_count = clinical_data['lymphocyte_count'].dropna()
    duct_count = clinical_data['duct_count'].dropna()
    duct_perimeter_avg = clinical_data['duct_perimeter_avg'].dropna()
    duct_size_avg = clinical_data['duct_size_avg'].dropna()

    # Only analyze if all data is present
    common_index = lymphocyte_count.index.intersection(duct_count.index).intersection(duct_perimeter_avg.index).intersection(duct_size_avg.index)
    lymphocyte_count = lymphocyte_count[common_index]
    duct_count = duct_count[common_index]
    duct_perimeter_avg = duct_perimeter_avg[common_index]
    duct_size_avg = duct_size_avg[common_index]

    # Calculate correlations
    corr_duct_count, p_duct_count = pearsonr(lymphocyte_count, duct_count)
    corr_duct_perimeter, p_duct_perimeter = pearsonr(lymphocyte_count, duct_perimeter_avg)
    corr_duct_size, p_duct_size = pearsonr(lymphocyte_count, duct_size_avg)
    
    print(f"Duct Count Correlation: {corr_duct_count:.3f}, p-value: {p_duct_count:.3e}")
    print(f"Duct Perimeter Average Correlation: {corr_duct_perimeter:.3f}, p-value: {p_duct_perimeter:.3e}")
    print(f"Duct Size Average Correlation: {corr_duct_size:.3f}, p-value: {p_duct_size:.3e}")
    

def ducts_plot(bool_data):
    # Histogram of duct count
    plt.figure(figsize=(4, 4))
    sns.histplot(
        clinical_data['duct_count'].dropna(),
        bins=100,
        kde=True,
        alpha=0.7
    )
    plt.title('Distribution of Duct Count')
    plt.xlabel('Duct Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("plots/ducts/distribution_ductcount.png", dpi=300)
    plt.close()

    # Scatter plot: duct count vs lymphocyte count
    plt.figure(figsize=(4, 4))
    sns.scatterplot(
        x=clinical_data['duct_count'],
        y=clinical_data['lymphocyte_count'],
        alpha=0.5
    )
    plt.title('Duct Count vs Lymphocyte Count')
    plt.xlabel('Duct Count')
    plt.ylabel('Lymphocyte Count')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("plots/ducts/ductcount_vs_lymphocytecount.png", dpi=300)
    plt.close()


def lymphocyte_boxplot(bool_data, target):
    bool_data = bool_data.copy()
    bool_data = bool_data[[col for col in bool_data.columns if col in ['age_diagnose', 'cox2_score', 'er', 'pr', 'her2', 'grade', 'p16_conclusion', 'p53_percentage', 'lymphocyte_count']]].dropna(subset=[target])
    bool_data['age_diagnose'] = bool_data['age_diagnose'].apply(lambda x: 0 if x < 50 else (1 if x >= 50 else np.nan))
    cols = [c for c in bool_data.columns if c != target]

    dfs = []
    for col in cols:
        temp = bool_data[[col, target]].dropna().copy()
        temp['category_value'] = temp[col].astype(str)
        temp['variable'] = col
        temp = temp[['variable', 'category_value', target]]
        dfs.append(temp)

    long_df = pd.concat(dfs, ignore_index=True)
    long_df['var_cat'] = long_df['variable'] + ': ' + long_df['category_value']

    long_df['category_num'] = long_df['category_value'].apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Sort by variable, then by numeric category ascending
    sorted_var_cat_order = long_df['var_cat'].unique().tolist()
    sorted_var_cat_order = ['age_diagnose: 0', 'age_diagnose: 1', 'grade: 0.0', 'grade: 1.0', 'er: 0.0', 'er: 1.0', 'pr: 0.0', 'pr: 1.0', 'her2: 0.0', 'her2: 1.0', 'p16_conclusion: 0.0', 'p16_conclusion: 1.0', 'cox2_score: 0.0', 'cox2_score: 1.0', 'p53_percentage: 0', 'p53_percentage: 1']

    plt.figure(figsize=(3, max(4, 0.3 * len(sorted_var_cat_order))))

    unique_vars = long_df['variable'].unique()
    var_palette = sns.color_palette("hls", n_colors=len(unique_vars))
    var_colors = dict(zip(unique_vars, var_palette))

    ax = sns.boxplot(
        x=target,
        y='var_cat',
        data=long_df,
        order=sorted_var_cat_order,
        orient='h',
        showcaps=True,
        boxprops=dict(facecolor='white', edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black'),
        flierprops=dict(markerfacecolor='none', markeredgecolor='black'),
        showfliers=False,
    )

    yticks = ax.get_yticks()
    ylabels = ax.get_yticklabels()
    label_to_y = {label.get_text(): pos for label, pos in zip(ylabels, yticks)}
    for var_cat, group in long_df.groupby('var_cat'):
        var = group['variable'].iloc[0]
        y = label_to_y[var_cat]
        x = group[target]
        ax.scatter(x, np.full_like(x, y), 
                color=var_colors[var], 
                edgecolors='none', 
                s=10, alpha=0.3,
                zorder=10)

    for var in unique_vars:
        sub_df = long_df[long_df['variable'] == var]
        groups = sub_df['var_cat'].unique()
        if len(groups) < 2:
            continue

        # Significance testing
        if len(groups) == 2:
            g1_vals = sub_df[sub_df['var_cat'] == groups[0]][target]
            g2_vals = sub_df[sub_df['var_cat'] == groups[1]][target]
            stat, pval = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
            print(f"{groups[0]} vs {groups[1]}: p = {pval:.4e}")

    plt.xlabel('Lymphocyte count')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'plots/boxplots/{target}_boxplots.png')


def effect_size(bool_data, target):
    bool_data = bool_data.dropna(subset=[target])

    significant_vars = [c for c in bool_data.columns if c != target]

    results = []

    # Conventional threshold for large effect sizes
    LARGE_EFFECT_THRESHOLDS = {
        'rank_biserial': 0.5,  # rule of thumb: 0.1 small, 0.3 medium, 0.5 large
        'eta_squared': 0.14    # rule of thumb: 0.01 small, 0.06 medium, 0.14 large
    }

    for col in significant_vars:
        temp = bool_data[[col, target]].dropna()
        values = temp[col].dropna().unique()
        
        if len(values) < 2:
            continue  # skip if only one group

        groups = [temp[temp[col] == val][target].values for val in values]

        if len(values) == 2:
            # Binary variable → Mann-Whitney + rank biserial
            group1, group2 = groups
            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            n1, n2 = len(group1), len(group2)
            u = stat
            rb = 1 - (2 * u) / (n1 * n2)  # Rank biserial correlation

            results.append({
                'variable': col,
                'type': 'binary',
                'test': 'Mann-Whitney U',
                'p_value': p,
                'effect_size': abs(rb),
                'effect_metric': 'rank_biserial',
                'large_effect': abs(rb) >= LARGE_EFFECT_THRESHOLDS['rank_biserial']
            })

        else:
            # Multi-class variable → Kruskal-Wallis + eta-squared
            stat, p = kruskal(*groups)
            all_values = np.concatenate(groups)
            eta_sq = stat * 1.0 / (len(all_values) - 1)

            results.append({
                'variable': col,
                'type': 'multi-class',
                'test': 'Kruskal-Wallis',
                'p_value': p,
                'effect_size': eta_sq,
                'effect_metric': 'eta_squared',
                'large_effect': eta_sq >= LARGE_EFFECT_THRESHOLDS['eta_squared']
            })

    # Compile results into a DataFrame
    effect_df = pd.DataFrame(results).sort_values('effect_size', ascending=False)

    significant_effects = effect_df[(effect_df['p_value'] < 0.05) & (effect_df['large_effect'])]

    print("All effect sizes:\n", effect_df)
    print("\nVariables with significant and large effects:\n", significant_effects)


def duct_associated_area(bool_data):
    bool_data['duct_lymphocyte_ratio'] = bool_data['lymphocyte_count'] * (bool_data['ducts_poly_area_mm2'] - bool_data['duct_size_total_mm2'])
    return bool_data


def multiple_testing(results, method):
    cols = list(results.keys())
    pvals = [results[col]['pval'] for col in cols]

    # Adjust p-values
    reject, pvals_corrected, _, _ = multipletests(pvals, method=method)

    for col, adj_p, rej in zip(cols, pvals_corrected, reject):
        results[col]['pval_adj'] = adj_p
        results[col]['reject_null'] = rej

    return results


def univariate_linear_analysis(data, target_column):
    results = {}

    drop_columns = [
        'event_months',
        'radiotherapy',
        'lymphocyte_count',
        'avg_lymphocyte_density',
        'lymph_density_var',
        'lymph_density_entropy',
        'lymph_density_moran_raw',
        'lymph_density_moran_hotspot',
        'lymphocyte_area_mm2',
        'pathologist_tilsscore',
        'duct_count',
        'duct_perimeter_avg',
        'duct_size_avg_mm2',
        'duct_size_total_mm2',
        'ducts_poly_area_mm2',
        'duct_lymphocyte_ratio',
        'duct_associated_area_mm2',
        'TLS',
        'TLS_count',
        'TLS_area',
        'first_subseq_event'
    ]

    data = data.drop(columns=[col for col in drop_columns if col != target_column and col in data.columns])
    df_clean = data[[target_column] + [col for col in data.columns if col != target_column]].dropna()

    y = df_clean[target_column]
    if (y <= 0).any():
        shift = abs(y.min()) + 1
        y = np.log(y + shift)
    else:
        y = np.log(y)
    y = StandardScaler().fit_transform(y.values.reshape(-1, 1)).flatten()

    for col in df_clean.columns:
        if col == target_column:
            continue
        
        X = df_clean[[col]]
        X_scaled = StandardScaler().fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=[col], index=X.index) 
        X_scaled = sm.add_constant(X_scaled)

        model = sm.OLS(y, X_scaled).fit()

        results[col] = {
            'coef': model.params[col],
            'pval': model.pvalues[col],
            'conf_int': model.conf_int().loc[col].tolist()
        }

    # Adjust for multiple testing
    results = multiple_testing(results, method="fdr_bh")

    return results, df_clean


def univariate_logistic_analysis(data, target_column):
    results = {}

    drop_columns = [
        'age_diagnose', 'radiotherapy', 'event_months', 'pathologist_tilsscore', 'grade',
        'er', 'pr', 'her2', 'p16_conclusion', 'p53_percentage', 'cox2_score'
    ]
    
    data = data.drop(columns=[col for col in drop_columns if col != target_column and col in data.columns])
    df_clean = data[[target_column] + [col for col in data.columns if col != target_column]].dropna()

    y = df_clean[target_column].astype(int)

    for col in df_clean.columns:
        if col == target_column:
            continue

        X = df_clean[[col]].copy()

        # Standardize if numeric and > 2 unique values
        if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 2:
            X[col] = StandardScaler().fit_transform(X[[col]])

        X = sm.add_constant(X)

        try:
            model = sm.Logit(y, X).fit(disp=False)
            coef = model.params[col]
            pval = model.pvalues[col]
            conf_int = model.conf_int().loc[col].tolist()
            results[col] = {
                'coef': coef,
                'pval': pval,
                'conf_int': conf_int
            }
        except Exception as e:
            print(f"Skipped {col} due to error: {e}")

    # Adjust for multiple testing
    results = multiple_testing(results, method="fdr_bh")

    return results, df_clean


def plot_univariate_results(results, df_clean, target_column):
    # Plot in column order
    col_order = [col for col in df_clean.columns if col != target_column and col in results]
    df = pd.DataFrame(results).T.loc[col_order]

    plt.figure(figsize=(8, len(df) * 0.4))
    sns.pointplot(x='coef', y=df.index, data=df, join=False)
    
    ci = np.array(df['conf_int'].tolist())
    plt.errorbar(df['coef'], df.index, 
                 xerr=[df['coef'] - ci[:,0], ci[:,1] - df['coef']],
                 fmt='none', c='gray')

    def pval_sig(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else: return 'ns'

    df['sig'] = df['pval_adj'].apply(pval_sig)

    plt.figure(figsize=(4, len(df) * 0.4))

    # Point plot with hue for significance
    sns.pointplot(
        x='coef',
        y=df.index,
        data=df,
        join=False,
        hue='sig',
        palette={
            '***': '#1a1aff',    # bright blue
            '**': '#00b140',     # vivid green
            '*': '#ff6f00',      # strong orange
            'ns': '#888888'      # grey
        },
        dodge=False,
        legend=False
    )

    # Confidence intervals
    ci = np.array(df['conf_int'].tolist())
    plt.errorbar(df['coef'], df.index,
                 xerr=[df['coef'] - ci[:, 0], ci[:, 1] - df['coef']],
                 fmt='none', c='gray')

    sig_palette = {
        '***': '#1a1aff',    # bright blue
        '**': '#00b140',     # vivid green
        '*': '#ff6f00',      # strong orange
        'ns': '#888888'      # grey
    }
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='*** (p < 0.001)', markerfacecolor=sig_palette['***'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='** (p < 0.01)', markerfacecolor=sig_palette['**'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='* (p < 0.05)', markerfacecolor=sig_palette['*'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='ns (p ≥ 0.05)', markerfacecolor=sig_palette['ns'], markersize=8),
    ]
    plt.legend(handles=legend_elements, title='Significance', loc='best', frameon=True)

    plt.axvline(0, color='red', linestyle='--')
    plt.title(f'Univariate Regression Coefficients: {target_column}')
    plt.xlabel('Standardized Effect Size')
    plt.ylabel('Clinical variable')  # Set y-axis label explicitly
    plt.tight_layout()
    plt.savefig(f"plots/univariate_TLS/univariate_{target_column}2.png", dpi=300)


def multivariable_analysis(bool_data):
    df = bool_data.copy()
    df = df.drop(columns=['avg_lymphocyte_density', 'event_months', 'pathologist_tilsscore', 'radiotherapy', 'TLS_count', 'p16_conclusion', 'pr', 'er', 'grade', 'her2', 'p53_percentage', 'cox2_score', 'TLS'])

    X = df.drop(columns=["first_subseq_event"])
    y = df["first_subseq_event"]

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Define numerical columns
    num_cols = [col for col in X.select_dtypes(include=["float64", "int64"]).columns if col != "first_subseq_event"]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), num_cols)
        # ("bin", SimpleImputer(strategy="most_frequent"), binary_cols),
    ])

    # LightGBM classifier
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass" if y.nunique() > 2 else "binary",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgb_model)
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Evaluation
    print("Classification Report:")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_refined.png", bbox_inches='tight')

    # ROC-AUC
    roc_auc = roc_auc_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)

    print(f"ROC AUC: {roc_auc:.3f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"LightGBM (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve_refined.png", bbox_inches='tight')

    # Fit pipeline
    pipeline.fit(X, y)

    # Feature importance Plot
    model = pipeline.named_steps["classifier"]
    importances = model.booster_.feature_importance(importance_type='gain')

    # Plot
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (LightGBM)")
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), [num_cols[i] for i in sorted_idx], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("feature_importances_refined.png", bbox_inches='tight')


def lymph_infiltrate(bool_data, infiltrate_metric, target=None):
    data = bool_data.dropna(subset=[infiltrate_metric, 'first_subseq_event']).copy()
    data = data[data['radiotherapy'] == 0] # Drop radiotherapy
    
    label = f"infiltrate_{infiltrate_metric}"
    filename = f"plots/clinical/median/MEDIAN_infiltrate_{infiltrate_metric}.png"

    if target:
        if target not in data.columns:
            raise ValueError(f"Target '{target}' not in dataframe.")
        if data[target].nunique() < 2:
            print(f"Not enough target variation in data for '{target}'")
            return
        filename = f"plots/clinical/median/FINAL_{target}_vs_infiltrate_{infiltrate_metric}.png"

    # Use median as a simple threshold
    optimal_threshold = data[infiltrate_metric].median()

    # Define high vs low infiltrate
    data[label] = np.where(data[infiltrate_metric] >= optimal_threshold, 1, 0)

    # In case of subgroup (target) analysis
    if target:
        for target_group, group_df in data.groupby(target):
            ct = pd.crosstab(group_df[label], group_df['first_subseq_event']).reindex(index=[0,1], columns=[0,1], fill_value=0)
            if ct.shape != (2, 2):
                print(f"Skipping Fisher's test for target={target_group}: unexpected table shape {ct.shape}")
                continue
            oddsratio, p_value = fisher_exact(ct)
            print(ct)
            print(f"\nFisher's Exact Test for {label} (0 vs 1) and recurrence within {target}={target_group}:")
            print(f"Odds Ratio = {oddsratio:.2f}")
            print(f"P-value = {p_value:.4f}")

        grouped = data.groupby([target, label])['first_subseq_event']
        summary = data.groupby([label, target])['first_subseq_event'].agg(['sum', 'count']).reset_index()
        summary['Infiltrate'] = summary[label].map({0: 'Low', 1: 'High'})
        summary['Target'] = summary[target].map({0: f"{target}=0", 1: f"{target}=1"})
        total_counts = summary.groupby('Target')['count'].transform('sum')
        summary['Proportion'] = summary['sum'] / total_counts * 100

        plt.figure(figsize=(3, 3))
        sns.barplot(
            data=summary,
            x='Target',
            y='Proportion',
            hue='Infiltrate',
            palette='Reds',
            order=[f"{target}=0", f"{target}=1"],
            hue_order=['Low', 'High']
        )

        # Plot text above each bar
        for i, row in summary.iterrows():
            x = [f"{target}=0", f"{target}=1"].index(row['Target'])
            hue = 0 if row['Infiltrate'] == 'Low' else 1
            n_hue = summary['Infiltrate'].nunique()
            bar_width = 0.8 / n_hue
            xpos = x - 0.4 + bar_width / 2 + hue * bar_width
            plt.text(
                xpos,
                row['Proportion'],
                f"{row['Proportion']:.1f}%\n(n={int(row['sum'])})",
                ha='center',
                va='bottom',
                fontsize=8
            )

        plt.xticks(ticks=[0, 1], labels=['0', '1'])
        plt.ylabel('% Recurrence')
        plt.xlabel(target.capitalize())
        plt.title(f"Recurrence by {target} and {infiltrate_metric}")
        plt.ylim(0, 30)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    # In case of analysis of all data
    else:
        ct = pd.crosstab(data[label], data['first_subseq_event'])
        if ct.shape != (2, 2):
            print(f"Skipping Fisher's test: unexpected table shape {ct.shape}")
            return

        oddsratio, p_value = fisher_exact(ct)
        print(f"\nFisher's Exact Test for all patients and {infiltrate_metric}:")
        print(f"Odds Ratio = {oddsratio:.2f}")
        print(f"P-value = {p_value:.4f}")

        recurrence = ct[1]
        total = ct.sum(axis=1)
        proportion = recurrence / total * 100  # Convert to percentage

        plot_df = pd.DataFrame({
            'Group': ['Low infiltrate', 'High infiltrate'],
            'Proportion': proportion.values,
            'n': total.values
        })

        plt.figure(figsize=(3, 2))
        sns.barplot(data=plot_df, x='Group', y='Proportion', palette='Reds')
        plt.ylabel('% Recurrence')
        plt.title(f'Recurrence proportion by {infiltrate_metric}')
        plt.ylim(0, 25) 

        for i, row in plot_df.iterrows():
            plt.text(i, row['Proportion'] + 1,
             f"{row['Proportion']:.1f}%\n(n={int(row['n'])})",
             ha='center', fontsize=7)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


def total_recurrence_plot(bool_data):
    bool_data = bool_data[bool_data['radiotherapy'] == 0]
    variables = ['grade', 'her2', 'er', 'pr']
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4), sharey=True)

    for i, var in enumerate(variables):
        ax = axes[i] if len(variables) > 1 else axes
        # Only include rows where both var and lymphocyte_count are not null
        filtered = bool_data.dropna(subset=[var, 'lymphocyte_count', 'first_subseq_event'])
        # Calculate recurrence proportion for each group (0/1)
        group = filtered.groupby(var)['first_subseq_event'].agg(['mean', 'count', 'sum'])
        print(f'{var}: {group}')
        group = group.loc[[0, 1]] if set([0, 1]).issubset(group.index) else group
        bars = ax.bar(group.index.astype(str), group['mean'], color=['#b3cde3', '#005b96'])
        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        for idx, bar in enumerate(bars):
            n = group['count'].iloc[idx]
            rec = int(group['sum'].iloc[idx])
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.1%}\n(n={n}, rec={rec})",
                    ha='center', va='bottom', fontsize=8)
    axes[0].set_ylabel('Recurrence proportion')
    plt.tight_layout()
    plt.savefig("plots/recurrence_barplot.png", dpi=300)
    plt.close()


def main():
    ######### Use only to change/load clinical data for the first time
    clinical_data_path = "/data/groups/aiforoncology/archive/pathology/PRECISION/Precision_NKI_89_05/Block1/metadata/20230515_IRBd22-110_clinical-data.xlsx"
    clinical_data = load_data(clinical_data_path)
    bool_data = binarize_data(clinical_data)

    ######### Use only when reading clinical_data from memory
    clinical_data = pd.read_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/analysis/processed_clinical_data.csv")
    bool_data = pd.read_csv("/home/t.leppers/dev/pathology-projects/EllogonTILs/analysis/processed_bool_data.csv")

    # General analyses
    clinicaltableoverview(clinical_data)
    correlation_matrix(bool_data)
    duct_lymphocyte_correlation(bool_data)
    ducts_plot(bool_data)
    duct_associated_area(bool_data)

    # Immune-related analyses
    immune_related_metrics = [
        'lymphocyte_count',
        'avg_lymphocyte_density',
        'lymph_density_var',
        'lymph_density_entropy',
        'lymph_density_moran_raw',
        'lymph_density_moran_hotspot',
        'duct_count',
        'duct_perimeter_avg',
        'duct_size_avg_mm2',
        'duct_size_total_mm2',
        'duct_associated_area_mm2',
        'duct_lymphocyte_ratio',
        'duct_associated_area_mm2',
        'TLS',
        'TLS_count',
        'TLS_area',
    ]

    for target in immune_related_metrics:
        lymphocyte_boxplot(bool_data, target=target)
        effect_size(bool_data, target=target)

        # Univariate analysis of immune-related metrics
        results, df_clean = univariate_linear_analysis(bool_data, target)
        plot_univariate_results(results, df_clean, target)

        # Lymphocytic infiltrate vs recurrence
        lymph_infiltrate(bool_data, infiltrate_metric=target)

    # Univariate analysis of first subsequence event
    results, df_clean = univariate_logistic_analysis(bool_data, target_column='first_subseq_event')
    plot_univariate_results(results, df_clean, target_column='first_subseq_event')

    # LightGBM (multivariable analysis predicting recurrence)
    multivariable_analysis(bool_data)

    # Lymphocytic infiltrate vs recurrence (subgroup analysis)
    for subgroup in ['her2', 'grade', 'er', 'pr']:
        for metric in immune_related_metrics:
            lymph_infiltrate(bool_data, infiltrate_metric=metric, target=subgroup)

    total_recurrence_plot(bool_data)


if __name__ == "__main__":
    main()
