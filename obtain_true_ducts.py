import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import openslide
import alphashape
from shapely.geometry import box, Polygon, MultiPoint
from shapely.ops import unary_union
import json


def load_data(data_path):
    with open(data_path, 'r') as file:
        data = pd.read_csv(file)

    data = data[data['object_check'] == True] # Remove false ducts
    data['Area_mm2'] = data['Area'] * 1e-6  # Convert area from um^2 to mm^2
    return data


def overview_plot(data):
    data = data.copy()
    data = data.dropna(subset=['Perimeter', 'Area_mm2'])  # Drop missing values
    data['log_Perimeter'] = np.log(data['Perimeter'])
    data['log_Area_mm2'] = np.log(data['Area_mm2'])

    plt.figure(figsize=(5, 4))

    plt.subplot(1, 2, 1)
    palette = sns.color_palette("muted")
    sns.violinplot(y=data['log_Perimeter'], color=palette[3], linewidth=1.5, edgecolor='black')
    # sns.swarmplot(y=data['log_Perimeter'], color='black', size=2, alpha=0.6)
    plt.title('Violinplot of Duct Perimeter')
    plt.ylabel('log(perimeter)')

    plt.subplot(1, 2, 2)
    sns.violinplot(y=data['log_Area_mm2'], color=palette[1], linewidth=1.5, edgecolor='black')
    # sns.swarmplot(y=data['log_Area_mm2'], color='black', size=2, alpha=0.6)
    plt.title('Violinplot of Duct Area (mm²)')
    plt.ylabel('log(area (mm²))')

    plt.tight_layout()
    plt.savefig('duct_violinplots.png')
    plt.close()


def substract_info(data):
    data = data.copy()
    data['T_number_split'] = data['T.nummer'].apply(lambda x: str(x).split("_")[0])
    grouped = data.groupby('T_number_split')
    duct_count = grouped.size().to_dict()
    duct_size_avg = grouped['Area_mm2'].mean().to_dict()
    duct_size_total = grouped['Area_mm2'].sum().to_dict()
    duct_perimeter_avg = grouped['Perimeter'].mean().to_dict()

    with open('duct_count.pkl', 'wb') as f:
        pd.to_pickle(duct_count, f)

    with open('duct_size_avg.pkl', 'wb') as f:
        pd.to_pickle(duct_size_avg, f)
        
    with open('duct_size_total.pkl', 'wb') as f:
        pd.to_pickle(duct_size_total, f)

    with open('duct_perimeter_avg.pkl', 'wb') as f:
        pd.to_pickle(duct_perimeter_avg, f)


def duct_associated_area(group, buffer_size):
    boxes = []
    for _, row in group.iterrows():
        xmin, xmax = row['XMin'], row['XMax']
        ymin, ymax = row['YMin'], row['YMax']
        boxes += [
            (xmin, ymin), (xmin, ymax),
            (xmax, ymin), (xmax, ymax)
        ]

    polygon = MultiPoint(boxes).convex_hull.buffer(buffer_size)

    return polygon


def duct_associated_area_concave(group):
    # Collect all corners of the bounding boxes (not just centers!)
    points = []
    for _, row in group.iterrows():
        xmin, xmax = row['XMin'], row['XMax']
        ymin, ymax = row['YMin'], row['YMax']
        points += [
            (xmin, ymin), (xmin, ymax),
            (xmax, ymin), (xmax, ymax)
        ]

    alpha = alphashape.optimizealpha(points)
    shape = alphashape.alphashape(points, alpha)

    print(shape)


def overlay_annotations(group,  tnumber, wsidir, polygon):
    tnumber = tnumber.replace('-', '_')
    wsi_path = next((os.path.join(wsidir, f) for f in os.listdir(wsidir) if tnumber in f and f.endswith('.mrxs')), None)
    wsi = openslide.OpenSlide(wsi_path)

    # Low resolution
    level = wsi.get_best_level_for_downsample(32)
    img = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB")
    downsample = wsi.level_downsamples[level]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    polys = [polygon] if isinstance(polygon, Polygon) else polygon.geoms
    for poly in polys:
        x, y = poly.exterior.xy
        x_scaled = [scale(xx, downsample) for xx in x]
        y_scaled = [scale(yy, downsample) for yy in y]
        ax.plot(x_scaled, y_scaled, color='gold', linewidth=2)

    for _, row in group.iterrows():
        xmin, xmax = scale(row['XMin'], downsample), scale(row['XMax'], downsample)
        ymin, ymax = scale(row['YMin'], downsample), scale(row['YMax'], downsample)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    ax.set_title(f"t-number: {tnumber}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"tight_{tnumber}_ducts_overlay.png")


def scale(coord, downsample):
    return int(coord / downsample)


def area_to_mm2(polygon, mppx=0.242534722222222, mppy=0.242647058823529):
    return polygon.area * mppx * mppy / 1e6


def main():
    data_path = "/home/t.leppers/ducts/true_ducts_marcelo.csv"
    data = load_data(data_path)
    wsi_dir = "/data/groups/aiforoncology/archive/pathology/PRECISION/Precision_NKI_89_05/Block1/datasets/P1000/images/"
    overview_plot(data)
    substract_info(data)

    results = []
    for tnumber in data['T.nummer'].unique():
        data['T.nummer'] = data['T.nummer'].astype(str)
        group = data[data['T.nummer'] == tnumber]
        polygon = duct_associated_area(group, buffer_size=3000)
        polygon = duct_associated_area_concave(group)
        polygon_area = area_to_mm2(polygon)
        results.append({'tnumber': tnumber, 'duct_associated_area': polygon_area})
        overlay_annotations(group, tnumber, wsi_dir, polygon)

    df = pd.DataFrame(results)
    df.to_csv('duct_associated_area.csv', index=False)


if __name__ == "__main__":
    main()
