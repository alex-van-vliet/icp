from pathlib import Path
import numpy as np
from skimage import io

def find(data):
    start = 0
    while start < data.shape[0] and data[start]:
        start += 1
    end = data.shape[0] - 1
    while end >= start and data[end]:
        end -= 1
    end += 1
    return start, end

def clean_borders(path):
    image = io.imread(path)
    whites = (image[:, :, 0] == 255) & (image[:, :, 1] == 255) & (image[:, :, 2] == 255)
    columns = np.sum(whites, axis=0) == image.shape[0]
    start_col, end_col = find(columns)
    rows = np.sum(whites, axis=1) == image.shape[1]
    start_row, end_row = find(rows)
    
    io.imsave(path, image[start_row:end_row, start_col:end_col])

def main():
    results = Path("results")
    
    for image in results.glob("**/*.png"):
        clean_borders(image)

if __name__ == '__main__':
    main()
