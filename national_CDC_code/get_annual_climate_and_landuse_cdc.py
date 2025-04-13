import pandas as pd
import xarray as xr
import cv2
import numpy as np
from pathlib import Path

# Set relative data directory path
DATA_DIR = Path(__file__).parent / "data_demo"
### alert: the Climate data path is downloaded from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS) and is not included in this repository.
### please register and download the data
CLIMATE_DATA = Path("../../WNV_project_files/WNV/climate/weather_data_to_be_downloaded.nc").resolve()

## alert: the land use data path is downloaded from "https://www.earthenv.org/landcover" and is not included in this repository. (Total 12 images)
LAND_USE_DIR = Path("../../WNV_project_files/WNV/climate/consensus_land_cover_data").resolve()

# Load CDC dataset
cdc_df = pd.read_csv(DATA_DIR / "cdc_demo.csv")

# Drop rows with missing latitude or longitude
cdc_df = cdc_df.dropna(subset=["Longitude", "Latitude"]).reset_index(drop=True)

# Add a 'Date' column for annual climate lookup
cdc_df["Date"] = pd.to_datetime(cdc_df["Year"].astype(str) + "-01-01")

# Load and sort ERA5-Land monthly dataset
print("Start adding climate variables")
ds = xr.open_dataset(CLIMATE_DATA).sortby("time")

# Convert columns to DataArrays for selection
latitude_da = xr.DataArray(cdc_df["Latitude"].values, dims="county")
longitude_da = xr.DataArray(cdc_df["Longitude"].values, dims="county")
time_da = xr.DataArray(cdc_df["Date"].values, dims="county")
year_da = xr.DataArray(cdc_df["Year"].values, dims="county")

# Define climate variables to extract
variable_list = [
    "u10", "v10", "t2m", "lai_hv", "lai_lv", "src", "sf",
    "ssr", "sro", "e", "tp", "swvl1"
]

# Compute annual weighted average
def weighted_temporal_mean(ds, var):
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    obs = ds[var]
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")
    return obs_sum / ones_out

# Add annual climate averages
for variable in variable_list:
    print("Start", variable)
    annual_variable = weighted_temporal_mean(ds, variable)
    average_variable = annual_variable.sel(latitude=latitude_da, longitude=longitude_da, time=time_da, expver=1, method="nearest")
    cdc_df["avg_" + variable] = average_variable.values
    average_variable.close()
    print("Finish", variable)

# Add climate max/min values per year
for variable in variable_list:
    print("Start", variable)
    max_variable = ds[variable].groupby("time.year").max(dim="time")
    min_variable = ds[variable].groupby("time.year").min(dim="time")
    max_sel = max_variable.sel(latitude=latitude_da, longitude=longitude_da, year=year_da, expver=1, method="nearest")
    min_sel = min_variable.sel(latitude=latitude_da, longitude=longitude_da, year=year_da, expver=1, method="nearest")
    cdc_df["max_" + variable] = max_sel.values
    cdc_df["min_" + variable] = min_sel.values
    max_sel.close()
    min_sel.close()
    print("Finish", variable)

# Load consensus land use images
print("Start adding land use")
images = [cv2.imread(str(LAND_USE_DIR / f"consensus_full_class_{i}.tif"))[:, :, 0].copy() for i in range(1, 13)]

# Convert images into xarray DataArrays with lat/lon coordinates
dataset = [
    xr.DataArray(
        im,
        coords=[np.linspace(90, -56, im.shape[0]), np.linspace(-180, 180, im.shape[1])],
        dims=["latitude", "longitude"],
    ) for im in images
]

# Assemble into a named Dataset
land_use_names = [
    "Evergreen/Deciduous Needleleaf Trees", "Evergreen Broadleaf Trees", "Deciduous Broadleaf Trees",
    "Mixed Trees", "Shrub", "Herbaceous", "Culture/Managed", "Wetland",
    "Urban/Built", "Snow/Ice", "Barren", "Water"
]

dataset = xr.Dataset(dict(zip(land_use_names, dataset)))

# Add land use values to DataFrame
for land_use in land_use_names:
    land_use_da = dataset[land_use].sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
    cdc_df[land_use] = land_use_da.values

dataset.close()

# Save final DataFrame
cdc_df.to_csv(DATA_DIR / "cdc_demoe_with_climate.csv", index=False)