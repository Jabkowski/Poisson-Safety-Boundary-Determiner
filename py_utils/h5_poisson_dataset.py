import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class H5PoissonDataset(Dataset):
    """
    Klasa Dataset wczytująca wszystkie wygenerowane mapy i powiązane rozkłady h
    z pliku HDF5 bezpośrednio do pamięci RAM, dla maksymalnej wydajności i stabilności na GPU.
    """

    def __init__(self, file_path):
        self.grids = []
        self.h_values = []
        self.dhdx_values = []
        self.dhdy_values = []
        self.ux_values = []
        self.uy_values = []
        self.normalization_stats = None

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Nie znaleziono pliku bazy danych: {file_path}"
            )

        print(f"Ładowanie zestawu danych z pliku: {file_path}...")
        with h5py.File(file_path, "r") as f:
            # Sortujemy klucze (indeksy map) dla zachowania spójności
            keys = sorted(list(f["grid"].keys()))
            for key in tqdm(keys, desc="Wczytywanie map"):
                # Obrazy w bazie są zapisane jako [H, W] lub [1, H, W];
                # normalizujemy do [1, H, W], bo model oczekuje wejścia 4D [B, C, H, W].
                def _as_channel_tensor(data):
                    arr = np.asarray(data, dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    return torch.tensor(arr, dtype=torch.float32)

                grid_data = _as_channel_tensor(f["grid"][key][:])
                h_data = _as_channel_tensor(f["h"][key][:])
                dhdx_data = _as_channel_tensor(f["dhdx"][key][:])
                dhdy_data = _as_channel_tensor(f["dhdy"][key][:])
                ux_data = _as_channel_tensor(f["u_x"][key][:])
                uy_data = _as_channel_tensor(f["u_y"][key][:])

                self.grids.append(grid_data)
                self.h_values.append(h_data)
                self.dhdx_values.append(dhdx_data)
                self.dhdy_values.append(dhdy_data)
                self.ux_values.append(ux_data)
                self.uy_values.append(uy_data)

    def set_normalization_stats(self, normalization_stats):
        self.normalization_stats = normalization_stats

    def _normalize_field(self, field_name, tensor):
        if self.normalization_stats is None:
            return tensor
        if field_name not in self.normalization_stats:
            return tensor
        mean = self.normalization_stats[field_name]["mean"]
        std = self.normalization_stats[field_name]["std"]
        return (tensor - mean) / std

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return (
            self.grids[idx],
            self._normalize_field("h", self.h_values[idx]),
            self._normalize_field("dhdx", self.dhdx_values[idx]),
            self._normalize_field("dhdy", self.dhdy_values[idx]),
            self._normalize_field("ux", self.ux_values[idx]),
            self._normalize_field("uy", self.uy_values[idx]),
        )
