import geopandas as gpd
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle
import os
from esda.moran import Moran
from libpysal.weights import lat2W


def load_data(ductsfile, tilsfile):
    ducts = gpd.read_file(ductsfile)
    ducts.set_crs(None, inplace=True, allow_override=True)

    rawtils = pd.read_csv(tilsfile)["answer"].str.replace("'", '"')
    tils = rawtils.apply(json.loads)  # Convert json strings in "answer" column to dictionaries
    tils = [item for sublist in tils for item in sublist] # Flatten points
    tils_coords = np.array([[point["x"], point["y"]] for point in tils])
    
    return ducts, tils_coords


def load_tils(tilsfile):
    rawtils = pd.read_csv(tilsfile)["answer"].str.replace("'", '"')
    tils = rawtils.apply(json.loads)  # Convert json strings in "answer" column to dictionaries
    tils = [item for sublist in tils for item in sublist] # Flatten points
    tils_coords = np.array([[point["x"], point["y"]] for point in tils])

    return tils_coords


def calculate_density(imname, tils_coords):
    kde = gaussian_kde(tils_coords.T)
    x_min, y_min = tils_coords.min(axis=0)
    x_max, y_max = tils_coords.max(axis=0)
    x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    density = np.reshape(kde(positions).T, x.shape)
    density_values = kde(tils_coords.T) # Outputs density value for each coordinate
    density_max = np.max(density_values, axis=0) # Average density value per image
    density_75 = np.percentile(density_values, 75) # 75th percentile of density values

    #### Beware the plot is not perfectly overlaying actual image as the coordinates are not in the same space

    plt.imshow(np.rot90(density), extent=[x_min, x_max, y_min, y_max], cmap='inferno', origin='lower')
    # plt.scatter(tils_coords[:,0], tils_coords[:,1], s=2, color='red') # Uncomment to plot each coordinate
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Density')
    plt.savefig(f"densityplots/Density_plot_TILs_{imname}.png")
    plt.close()

    return density


def get_adaptive_threshold(density, lower_percentile=5):
    non_zero = density[density > 0]
    if non_zero.size == 0:
        return None
    return np.percentile(non_zero, lower_percentile)


def calculate_density_variance(imname, density, results):
    threshold = get_adaptive_threshold(density) # Filter out low density values (e.g. noise/no tissue)
    if threshold is None:
        print(f"No tissue found in {imname}.")
        results.append({"image_name": imname, "density_variance": np.nan})
        return

    filtered_density = density[density > threshold]
    densityvar = np.var(filtered_density)
    results.append({"image_name": imname, "density_variance": densityvar})


def calculate_density_entropy(imname, density, results):   
    threshold = get_adaptive_threshold(density) # Filter out low density values (e.g. noise/no tissue)
    if threshold is None:
        print(f"No tissue found in {imname}.")
        results.append({"image_name": imname, "entropy": np.nan})
        return

    filtered_density = density[density > threshold]
    filtered_density /= np.sum(filtered_density) # Normalize
    entropy = -np.sum(filtered_density * np.log(filtered_density))
    results.append({"image_name": imname, "entropy": entropy})


def calculate_density_moran(imname, density, raw_results, hotspots_results):
    threshold = get_adaptive_threshold(density) # Filter out low density values (e.g. noise/no tissue)
    if threshold is None:
        print(f"No tissue found in {imname}.")
        raw_results.append({"image_name": imname, "moran_I": np.nan})
        hotspots_results.append({"image_name": imname, "hotspot": np.nan})
        return
    
    filtered_density = np.where(density > threshold, density, 0) # Change low density values to 0
    raw_mi = Moran(filtered_density.flatten(), lat2W(100, 100))
    raw_results.append({
        "image_name": imname,
        "moran_I": raw_mi.I,
        "p_value": raw_mi.p_sim
    })

    hotspot_threshold = np.percentile(filtered_density, 90) # Define hotspot threshold as 90th percentile
    binary_density = (filtered_density >= hotspot_threshold).astype(int).flatten()
    hotspot_mi = Moran(binary_density, lat2W(100, 100))
    hotspots_results.append({
        "image_name": imname,
        "hotspot": hotspot_mi.I,
        "p_value": hotspot_mi.p_sim
    })


def statistics(all_densities):

    densities = np.array(list(all_densities.values()))

    # Describing the data
    print(f"Minimum density: {np.min(all_densities)}")
    print(f"Maximum density: {np.max(all_densities)}")
    print(f"Mean density: {np.mean(all_densities)}")
    print(f"Median density: {np.median(all_densities)}")
    print(f"Standard deviation: {np.std(all_densities)}")

    # Plotting the histogram
    plt.hist(all_densities, bins=50, color='blue')
    # Add standard deviation bars
    mean = np.mean(list(all_densities))
    std = np.std(list(all_densities))
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=1, label='+1 SD')
    plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=1, label='-1 SD')
    plt.legend()
    plt.xlabel('Density')
    plt.ylabel('Frequency')
    plt.title('Density Histogram over all images, per grid')
    plt.savefig("density/Density_histogram_per_grid.png")


def main():
    with open("/home/t.leppers/dev/pathology-projects/TLS/mappings/T_number_svs_mapping.json", "r") as f:
        mapping = json.load(f)

    variance_results = []
    entropy_results = []
    moran_raw_results = []
    hotspots_results = []
    filepath = "/projects/ellogon_tils/outputs/tjarda_dcis_v2.2.0/Aperio/"
    for file in os.listdir(filepath):
        tilsfile = os.path.join(filepath, file, "upload_slidescore.csv")
        if os.path.exists(tilsfile):
            mapping_filename = file.split("Aperio-images-")[-1]
            if mapping_filename in mapping:
                tnumber = mapping[mapping_filename]
                tils_coords = load_tils(tilsfile)
                density = calculate_density(tnumber, tils_coords)
                # calculate_density_variance(tnumber, density, variance_results)
                # calculate_density_entropy(tnumber, density, entropy_results)
                calculate_density_moran(tnumber, density, moran_raw_results, hotspots_results)
                print(f"Processed {tnumber}.")
    
    variance = pd.DataFrame(variance_results)
    entropy = pd.DataFrame(entropy_results)
    moran = pd.DataFrame(moran_raw_results)
    hotspots = pd.DataFrame(hotspots_results)

    variance.to_csv("density_variance.csv", index=False)
    entropy.to_csv("density_entropy.csv", index=False)
    moran.to_csv("density_moran_raw.csv", index=False)
    hotspots.to_csv("density_moran_hotspots.csv", index=False)


if __name__ == "__main__":
    main()
