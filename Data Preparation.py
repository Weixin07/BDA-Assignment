import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

## Step 1: Data Selection
# Load the tsunami data
tsunami_df = pd.read_csv('runups-2024-05-09_11-13-37_+0800.tsv', sep='\t')

# Select relevant columns
columns_of_interest = [
    'Year', 'Mo', 'Dy', 'Hr', 'Mn', 'Sec', 'Latitude', 'Longitude', 
    'Earthquake Magnitude', 'Max Water Height (m)', 'Distance From Source (km)'
]
tsunami_df = tsunami_df[columns_of_interest]

## Step 2: Data Cleaning
# Drop rows with missing values in critical columns
tsunami_df.dropna(subset=['Max Water Height (m)', 'Earthquake Magnitude', 'Distance From Source (km)', 'Latitude', 'Longitude'], inplace=True)

# Handling outliers using IQR for 'Max Water Height (m)'
Q1 = tsunami_df['Max Water Height (m)'].quantile(0.25)
Q3 = tsunami_df['Max Water Height (m)'].quantile(0.75)
IQR = Q3 - Q1
filter = (tsunami_df['Max Water Height (m)'] >= (Q1 - 1.5 * IQR)) & (tsunami_df['Max Water Height (m)'] <= (Q3 + 1.5 * IQR))
tsunami_df = tsunami_df[filter]

## Step 3: Integrate Data
# Load coastline data
coastline_df = gpd.read_file('ne_10m_coastline.shp')
# Convert to a more Japan-specific CRS, such as EPSG:2450 for general use across Japan
coastline_df = coastline_df.to_crs(epsg=2450)

# Convert tsunami data to GeoDataFrame
tsunami_gdf = gpd.GeoDataFrame(
    tsunami_df, geometry=gpd.points_from_xy(tsunami_df.Longitude, tsunami_df.Latitude)
)
# Set original CRS to WGS84 and convert to the same local CRS as coastline for accurate distance measurement
tsunami_gdf.set_crs(epsg=4326, inplace=True)  # WGS84
tsunami_gdf.to_crs(epsg=2450, inplace=True)  # Convert to a local CRS for Japan

# Calculate the nearest coastline for each tsunami event
tsunami_gdf['coastline_distance'] = tsunami_gdf.geometry.apply(
    lambda x: coastline_df.distance(x).min()
)

## Step 4: Format Data
# Fill NaN with default values or drop them before type conversion to avoid errors
tsunami_gdf['Year'].fillna(-1, inplace=True)  # -1 or some other sentinel value if NaNs exist
tsunami_gdf['Mo'].fillna(-1, inplace=True)
tsunami_gdf['Dy'].fillna(-1, inplace=True)

# Convert to integer types
tsunami_gdf['Year'] = tsunami_gdf['Year'].astype(int)
tsunami_gdf['Mo'] = tsunami_gdf['Mo'].astype(int)
tsunami_gdf['Dy'] = tsunami_gdf['Dy'].astype(int)

# Save or continue processing
tsunami_gdf.to_file("processed_data.gpkg", layer='tsunami', driver="GPKG")
