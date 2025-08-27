import numpy as np
import pandas as pd
import os
import json
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
from shapely.validation import explain_validity
import matplotlib.pyplot as plt


def extract_valid_polygons(geometry):
    valid_polys = []

    if geometry.geom_type == 'Polygon':
        geoms = [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        geoms = geometry.geoms
    else:
        return valid_polys

    for geom in geoms:
        if not geom.is_valid:
            print("Invalid geometry:", explain_validity(geom))
            try:
                line = LineString(geom.exterior.coords) # Use only exterior polygon
                polys = list(polygonize(line))
                valid_polys.extend(polys)
            except Exception as e:
                print(f"Polygonize failed: {e}")
        else:
            valid_polys.append(Polygon(geom.exterior))

    return valid_polys


def load_annotations(annotation):
    annotation_gdf = gpd.read_file(annotation)

    polygons = []
    for geometry in annotation_gdf.geometry:
        polygons.extend(extract_valid_polygons(geometry))

    return polygons


def compute_iou(polygon1, polygon2):
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    return intersection / union if union > 0 else 0


def plot_iou_stats(iou_results):
    # plot histogram of IOU values
    iou_values = iou_results['iou']
    plt.hist(iou_values, bins=20, edgecolor='black')
    plt.xlabel('IOU')
    plt.ylabel('Frequency')
    plt.title('Histogram of IOU values')
    plt.savefig('iou_histogram.png')


def evaluate_segmentation(gt_polygons, pred_polygons, iou_threshold=0.5):
    matched_gt = set()
    matched_pred = set()
    ious = []

    for i, gt in enumerate(gt_polygons):
        best_iou = 0
        best_j = -1
        for j, pred in enumerate(pred_polygons):
            if j in matched_pred:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold:
            matched_gt.add(i)
            matched_pred.add(best_j)
            ious.append(best_iou)

    TP = len(matched_gt)
    FN = len(gt_polygons) - TP
    FP = len(pred_polygons) - len(matched_pred)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = sum(ious) / TP if TP > 0 else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "all_ious": ious
    }

def accuracy_score(iou_results):
    ious = iou_results['iou']
    accuracy = sum(i >= 0.5 for i in ious) / len(ious) if len(ious) > 0 else 0
    mean_iou = np.mean(ious) if len(ious) > 0 else 0

    print('Accuracy: ', accuracy, 'Mean IOU: ', mean_iou)


def total_stats(metrics):
    total_TP = metrics['TP'].sum()
    total_FP = metrics['FP'].sum()
    total_FN = metrics['FN'].sum()
    total_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    print(f"Total TP: {total_TP}")
    print(f"Total FP: {total_FP}")
    print(f"Total FN: {total_FN}")
    print(f"Total predictions: {total_TP + total_FP}")
    print(f"Total ground truth: {total_TP + total_FN}")
    print(f"Precision: {total_precision:.3f}")
    print(f"Recall:    {total_recall:.3f}")
    print(f"F1 Score:  {total_f1:.3f}")


def main():
    # annotations_pathologist_dir = "/home/t.leppers/tls/annos_tls_marcos_Aperio" # In .json format
    # annotated_slides_dir = "/projects/dcis-recurrence-tjarda/Aperio_container_outputs/images/converted" # In .geojson format
    # iou_results = []
    # metrics_summary = []

    # for annotation in os.listdir(annotations_pathologist_dir):
    #     pathologist_annotation = os.path.join(annotations_pathologist_dir, annotation)
    #     imagename = annotation.split(".")[0]

    #     hooknet_annotation = os.path.join(annotated_slides_dir, imagename + ".geojson")
    #     if not os.path.exists(hooknet_annotation):
    #         print(f"HookNet annotation for {imagename} not found.")
    #         continue

    #     print(f"Processing {imagename}.")

    #     pathologist_polygons = load_annotations(pathologist_annotation)
    #     hooknet_polygons = load_annotations(hooknet_annotation)

    #     # Evaluate and match polygons
    #     stats = evaluate_segmentation(pathologist_polygons, hooknet_polygons, iou_threshold=0.5)

    #     # Log per-IoU result for histogram
    #     for iou in stats["all_ious"]:
    #         iou_results.append({
    #             "imagename": imagename,
    #             "iou": iou
    #         })

    #     # Store per-image metrics
    #     metrics_summary.append({
    #         "imagename": imagename,
    #         "TP": stats["TP"],
    #         "FP": stats["FP"],
    #         "FN": stats["FN"],
    #         "precision": stats["precision"],
    #         "recall": stats["recall"],
    #         "f1": stats["f1"],
    #         "mean_iou": stats["mean_iou"]
    #     })
        
    #     # spatial_index = STRtree(hooknet_polygons) # Create spatial index to find nearby or overlapping polygons
    #     # for pathologist_polygon in pathologist_polygons:
    #     #     nearby_indices = spatial_index.query(pathologist_polygon)
    #     #     nearby_polygons = [hooknet_polygons[i] for i in nearby_indices]
    #     #     if len(nearby_polygons) > 0:
    #     #         for hooknet_polygon in nearby_polygons:
    #     #             iou = compute_iou(pathologist_polygon, hooknet_polygon)
    #     #             print(f"IOU between {imagename} pathologist and hooknet annotations: {iou}\n")
    #     #             iou_results.append({
    #     #                 "imagename": imagename,
    #     #                 "iou": iou
    #     #             })
    #     #     else:
    #     #         print(f"No overlapping polygons found for {imagename} pathologist annotation.\n")
    #     #         iou_results.append({
    #     #             "imagename": imagename,
    #     #             "iou": 0.0
    #     #         })
    
    # # iou_results = pd.DataFrame(iou_results)
    # # iou_results.to_csv("iou_results.csv", index=False)
    # metrics_df = pd.DataFrame(metrics_summary)
    # metrics_df.to_csv("per_image_metrics.csv", index=False)

    # total_TP = metrics_df['TP'].sum()
    # total_FP = metrics_df['FP'].sum()
    # total_FN = metrics_df['FN'].sum()
    # total_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    # total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    # total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    # print("\n=== Overall Stats ===")
    # print(f"Total TP: {total_TP}")
    # print(f"Total FP: {total_FP}")
    # print(f"Total FN: {total_FN}")
    # print(f"Precision: {total_precision:.3f}")
    # print(f"Recall:    {total_recall:.3f}")
    # print(f"F1 Score:  {total_f1:.3f}")

    iou_results = pd.read_csv("iou_results.csv")
    # plot_iou_stats(iou_results)
    # accuracy_score(iou_results)
    per_image_metrics = pd.read_csv("per_image_metrics.csv")
    total_stats(per_image_metrics)


if __name__ == "__main__":
    main()
