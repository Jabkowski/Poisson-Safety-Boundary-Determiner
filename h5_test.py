import h5py


def print_tree(obj, indent=0):
    for key in obj:
        item = obj[key]
        if isinstance(item, h5py.Group):
            print("  " * indent + f"[Group] {key}/")
            print_tree(item, indent + 1)
        else:
            print(
                "  " * indent
                + f"[Dataset] {key} shape={item.shape} dtype={item.dtype}"
            )


with h5py.File("matlab/grids_data_512x512.h5", "r") as f:
    print_tree(f)
