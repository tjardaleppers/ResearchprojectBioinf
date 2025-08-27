import geopandas as gpd
import pandas as pd
import json
import math
from shapely.geometry import Point


"""
This script calculates the TILs score per DCIS duct by counting the number of TILs surrounding the duct
with different thresholds.
"""


# Location of TILs data: /projects/ellogon_tils/outputs/tjarda_dcis_v2.2.0/{image}/upload_slidescore.csv
# Location of ducts data: /data/groups/aiforoncology/derived/pathology/PRECISION/DCIS_duct_segmentations/geojsons
# Define mpp
mpp=0.5032


def load_data(ductsfile, tilsfile):
    ducts = gpd.read_file(ductsfile)
    rawtils = pd.read_csv(tilsfile)["answer"].str.replace("'", '"')
    tils = rawtils.apply(json.loads) # Convert json strings in "answer" column to dictionaries
    tils = pd.json_normalize(tils.explode())

    tils_gdf = gpd.GeoDataFrame(
        tils,
        geometry=[Point(xy) for xy in zip(tils["x"], tils["y"])]
        )

    tils_gdf.set_crs(None, inplace=True)
    ducts.set_crs(None, inplace=True, allow_override=True)

    return ducts, tils_gdf


def detect_lymphocytes(ducts, tils):
    """
    Detect lymphocytes per (buffered) ducts area.
    """

    lymphocyte = 8 / mpp # Average lymphocyte is 8 um
    thresholds = [lymphocyte, lymphocyte * 2, lymphocyte * 3, lymphocyte * 4] 

    ducts = ducts.copy() 
    ducts = ducts.drop(columns=["classification"])
    ducts["duct_area"] = ducts["geometry"].area * (mpp ** 2) # Insert new column with duct area in um^2
    tils = tils.copy()
    tils = gpd.sjoin(tils, ducts, predicate="within", how="left")
    tils = tils.drop(columns=["index_right"])
    lymphocyte_count = pd.DataFrame(index=ducts["id"])
    lymphocyte_count["lymphocytes_in_duct"] = tils.groupby("id").size() # Count number of lymphocytes per id
    lymphocyte_count["lymphocyte_area_no_buffer"] = lymphocyte_count["lymphocytes_in_duct"] * math.pi * (4 ** 2)

    for threshold in thresholds:
        ducts[f"buffer_{threshold}"] = ducts.geometry.buffer(threshold) # Expand duct area with threshold
        ducts[f"area_{threshold}"] = ducts[f"buffer_{threshold}"].area * (mpp ** 2) - ducts["duct_area"] # Calculate area of buffer
        tils_buffered = gpd.sjoin(tils, ducts.set_geometry(f"buffer_{threshold}"), predicate="within", how="left")
        lymphocyte_count[f"lymphocytes_in_duct_{threshold}"] = tils_buffered.groupby("id_right").size()
        lymphocyte_count[f"lymphocyte_area_duct_{threshold}"] = lymphocyte_count[f"lymphocytes_in_duct_{threshold}"] * math.pi * (4 ** 2)
        lymphocyte_count[f"lymphocyte_in_buffer_{threshold}"] = lymphocyte_count[f"lymphocytes_in_duct_{threshold}"] - lymphocyte_count["lymphocytes_in_duct"]
        lymphocyte_count[f"lymphocyte_area_buffer_{threshold}"] = lymphocyte_count[f"lymphocyte_in_buffer_{threshold}"] * math.pi * (4 ** 2)

    ducts_area = ducts[[column for column in ducts.columns if "area" in column]]
    lymphocyte_area = lymphocyte_count[[column for column in lymphocyte_count.columns if "area" in column]]
    
    return ducts_area, lymphocyte_area


def compute_tilsscore(ducts_area, lymphocyte_area, id):
    """
    Calculate the percentage of lymphocyte area in ducts area.
    """
    lymphocyte_area = lymphocyte_area[[column for column in lymphocyte_area.columns if "buffer" in column]]
    lymphocyte_area.columns = ducts_area.columns # Rename column names
    lymphocyte_area = lymphocyte_area.reset_index(drop=True)
    ducts_area = ducts_area.reset_index(drop=True)
    tilsscore = (lymphocyte_area / ducts_area) * 100
    tilsscore.dropna(how="all", inplace=True)
    tilsscore = tilsscore.mean(axis=0).to_frame().T
    tilsscore.index = [id]
    
    return tilsscore


def create_output(scores_df):
    scores_df.to_csv("/home/t.leppers/tils/tilsscores_per_duct_Marcelo.csv", index=True)


def main():
    tilsscores = pd.DataFrame()

    for line in filenames.itertuples(index=False):
        ductsfile = f"/data/groups/aiforoncology/derived/pathology/PRECISION/DCIS_duct_segmentations/geojsons/{line[1]}"
        tilsfile = f"/projects/ellogon_tils/outputs/tjarda_dcis_v2.2.0/Aperio-images-{line[0]}/upload_slidescore.csv"
        ducts, tils = load_data(ductsfile, tilsfile)
        ducts_area, lymphocyte_area = detect_lymphocytes(ducts, tils)
        tilsscore = compute_tilsscore(ducts_area, lymphocyte_area, id=line[0])
        tilsscores = pd.concat([tilsscores, tilsscore], axis=0)
    
    create_output(tilsscores)


if __name__ == "__main__":
    main()
