import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dl
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.unet import UNetBilinear
from py_utils.utils import (
    GridDataset,
    plot_validation_results,
    load_h5_to_torch,
    calc_grad_loss,
)


def TrainModel(model, train_loader, criterion, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for grid, h_true in pbar:
        grid = grid.to(device)
        h_true = h_true.to(device)

        optimizer.zero_grad()

        h_pred = model(grid)
        mse, loss_grad, loss_bc = calc_grad_loss(
            criterion, h_pred, h_true
        )

        # --- final ---
        loss = mse + loss_grad + 0.01 * loss_bc
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(
        f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.6f}"
    )


def ValidateModel(
    model,
    test_loader,
    criterion,
    h_std,
    h_mean,
    device,
    epoch_label="loaded",
    save_plot=False,
):
    model.eval()  # disables dropout, batchnorm
    test_loss = 0.0

    pbar = tqdm(test_loader, desc="Testing", leave=False)
    i = 0
    with torch.no_grad():
        for grid, h_true in pbar:
            grid = grid.to(device)
            h_true = h_true.to(device)

            h_pred = model(grid)

            mse, loss_grad, loss_bc = calc_grad_loss(
                criterion, h_pred, h_true
            )
            print(
                f"mse: {mse.item():.6f}, grad_loss: {loss_grad.item():.6f}, bc_loss: {loss_bc.item():.6f})"
            )
            test_loss += mse.item()
            pbar.set_postfix(loss=mse.item())
            plot_validation_results(
                h_pred, h_true, epoch_label, save_plot=save_plot, i=i
            )
            i += 1

    test_loss /= len(test_loader)
    print("Test MSE:", test_loss)


epochs = 15
batch_size = 4
WEIGHTS_PATH = (
    f"weights/grads_loss_v3_unetbilinear_model_e{epochs}.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetBilinear(in_channels=1, out_channels=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# load data from h5 file
grids, h = load_h5_to_torch("td.h5")

## Train-test split

grids_train, grids_test, h_train, h_test = train_test_split(
    grids, h, test_size=0.2, random_state=42
)

# normalize h only
h_mean = h_train.mean()
h_std = h_train.std() + 1e-8

h_train = (h_train - h_mean) / h_std
h_test = (h_test - h_mean) / h_std

train_dataset = GridDataset(grids_train, h_train)
test_dataset = GridDataset(grids_test, h_test)

train_loader = dl(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = dl(test_dataset, batch_size=batch_size, shuffle=False)
print(train_loader.__len__())
print(test_loader.__len__())

load_existing_weights = True

if load_existing_weights and os.path.isfile(WEIGHTS_PATH):
    model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=device)
    )
    print(f"Loaded existing model weights from {WEIGHTS_PATH}")
    ValidateModel(
        model,
        test_loader,
        criterion,
        h_std,
        h_mean,
        device,
        epoch_label="loaded",
        save_plot=True,
    )

else:
    print("h:", h_train.min().item(), h_train.max().item())
    for epoch in range(epochs):
        TrainModel(model, train_loader, criterion, device)

    ValidateModel(
        model,
        test_loader,
        criterion,
        h_std,
        h_mean,
        device,
        epoch_label=epochs,
        save_plot=True,
    )

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")
