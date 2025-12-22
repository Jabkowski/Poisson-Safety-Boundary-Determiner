import h5py
import numpy as np
import matplotlib.pyplot as plt
data = {}

def parse_tree(obj, indent=0, current=None):
    if current is None:
        current = data

    for key in obj:
        item = obj[key]
        if isinstance(item, h5py.Group):
            current[key] = {}
            parse_tree(item, indent + 1, current=current[key])
        else:
            # store dataset as a numpy array in the current group's dict
            current[key] = np.asarray(item[()])

def plot_data(types=None):
    if types is None:
        types = ["grid", "dhdx", "dhdy", "h"]

    grid_d = data[types[0]]
    dhdx_d = data[types[1]]
    dhdy_d = data[types[2]]
    h_d = data[types[3]]
    
    keys = list(grid_d.keys())
    arr_grid_d = grid_d[keys[0]]
    arr_dhdx_d = dhdx_d[keys[0]]
    arr_dhdy_d = dhdy_d[keys[0]]
    arr_h_d = h_d[keys[0]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(arr_grid_d)
    axes[0, 0].set_title("grid")
    axes[0, 1].imshow(arr_dhdx_d,cmap='gray')
    axes[0, 1].set_title("dhdx")
    axes[1, 0].imshow(arr_dhdy_d,cmap='gray')
    axes[1, 0].set_title("dhdy")
    axes[1, 1].imshow(arr_h_d)
    axes[1, 1].set_title("h")
    plt.tight_layout()
    plt.show()

with h5py.File("data/grids_data.h5", "r") as f:
    parse_tree(f)
    print("Collected groups:", list(data.keys()))
    plot_data(types=["grid", "dhdx", "dhdy", "h"])
    
    
    