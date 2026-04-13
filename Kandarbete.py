import geopandas as gpd
import numpy as np
import cv2
import random
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# Load file
gdf = gpd.read_file(r"C:\Users\Therese Johannesson\OneDrive - Uppsala universitet\Teknisk Fysik\År 3\Kandarbete\Trainlayers\Trainlayers\Test_Fagersta.gpkg")

# Choose output mask size
width, height = 1024, 1024

# Get bounds of all polygons
minx, miny, maxx, maxy = gdf.total_bounds

# Create transform
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# Rasterize polygons to mask
mask = rasterize(
    [(geom, 1) for geom in gdf.geometry if geom is not None],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# Rotate mask
angle = random.uniform(-15, 15)
h, w = mask.shape
M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

rotated_mask = cv2.warpAffine(
    mask,
    M,
    (w, h),
    flags=cv2.INTER_NEAREST,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)

# Save result
cv2.imwrite("mask.png", mask * 255)
cv2.imwrite("rotated_mask.png", rotated_mask * 255)

print("Done")
print("Rotation angle:", angle)
print("Unique values:", np.unique(rotated_mask))
