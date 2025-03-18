import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_channel_animation(output: torch.Tensor, batch_num: int = 0):
    """
    Creates an animation of all channels for a given batch over time.

    Parameters:
    - output (torch.Tensor): Input tensor of shape (Nbatch, Nchannel, Nx, Ny, Nt).
    - batch_num (int): The batch index to visualize (default is 0).

    Returns:
    - None (Displays an animation)
    """

    # Convert to NumPy if it's a PyTorch tensor
    if isinstance(output, torch.Tensor):
        output_np = output.numpy()
    else:
        output_np = output

    # Extract the selected batch
    Nbatch, Nchannel, Nx, Ny, Nt = output_np.shape
    if batch_num >= Nbatch or batch_num < 0:
        raise ValueError(f"Invalid batch number: {batch_num}. Must be between 0 and {Nbatch-1}.")

    first_batch = output_np[batch_num]  # Shape: (Nchannel, Nx, Ny, Nt)

    # Determine subplot layout
    nrows = int(np.ceil(np.sqrt(Nchannel)))  # Square-like arrangement
    ncols = int(np.ceil(Nchannel / nrows))

    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).flatten()  # Flatten to ensure it's iterable

    # Create initial images and colorbars
    ims = []
    for i in range(Nchannel):
        ax = axes[i]
        im = ax.imshow(first_batch[i, :, :, 0], cmap="viridis", animated=True)
        ax.set_title(f"Channel {i}")
        fig.colorbar(im, ax=ax, orientation="vertical")
        ims.append(im)

    # Hide extra subplots if Nchannel < grid size
    for j in range(Nchannel, len(axes)):
        axes[j].axis("off")  # Hide unused subplots

    # Animation update function
    def update(frame):
        for i in range(Nchannel):
            ims[i].set_array(first_batch[i, :, :, frame])
        return ims

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=100, blit=False)

    # Show animation
    plt.show()

def compare_tensors_animation(pred: torch.Tensor, gt: torch.Tensor, batch_num: int = 0):
    """
    Creates an animation comparing predicted and ground truth tensors.

    Parameters:
    - pred (torch.Tensor): Predicted tensor of shape (Nbatch, Nchannel, Nx, Ny, Nt).
    - gt (torch.Tensor): Ground truth tensor (same shape as pred).
    - batch_num (int): The batch index to visualize (default is 0).

    Returns:
    - None (Displays an animation)
    """

    # Convert to NumPy if tensors are PyTorch tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()

    # Validate shapes
    if pred.shape != gt.shape:
        raise ValueError("Predicted and ground truth tensors must have the same shape.")

    # Extract the selected batch
    Nbatch, Nchannel, Nx, Ny, Nt = pred.shape
    if batch_num >= Nbatch or batch_num < 0:
        raise ValueError(f"Invalid batch number: {batch_num}. Must be between 0 and {Nbatch-1}.")

    pred_batch = pred[batch_num]  # Shape: (Nchannel, Nx, Ny, Nt)
    gt_batch = gt[batch_num]  # Shape: (Nchannel, Nx, Ny, Nt)
    error_batch = np.abs(pred_batch - gt_batch)  # Absolute error

    # Determine subplot layout (3 plots per channel: GT, Pred, Error)
    nrows = Nchannel
    ncols = 3  # GT, Prediction, and Error

    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))

    ims = []
    for i in range(Nchannel):
        # Compute vmin/vmax per channel for correct color scaling
        vmin = np.min(gt_batch[i])  # Min value for this channel
        vmax = np.max(gt_batch[i])  # Max value for this channel

        # Plot Ground Truth
        ax = axes[i, 0]
        im_gt = ax.imshow(gt_batch[i, :, :, 0], cmap="viridis", animated=True, vmin=vmin, vmax=vmax)
        ax.set_title(f"GT - Channel {i}")
        cbar_gt = fig.colorbar(im_gt, ax=ax, orientation="vertical")  # Colorbar for GT

        # Plot Prediction (Using same vmin/vmax as GT)
        ax = axes[i, 1]
        im_pred = ax.imshow(pred_batch[i, :, :, 0], cmap="viridis", animated=True, vmin=vmin, vmax=vmax)
        ax.set_title(f"Pred - Channel {i}")
        cbar_pred = fig.colorbar(im_pred, ax=ax, orientation="vertical")  # Colorbar matches GT

        # Plot Error (Independent color scale)
        ax = axes[i, 2]
        im_err = ax.imshow(error_batch[i, :, :, 0], cmap="inferno", animated=True)  # Error in red shades
        ax.set_title(f"Error - Channel {i}")
        cbar_err = fig.colorbar(im_err, ax=ax, orientation="vertical")  # Independent colorbar for error

        ims.append((im_gt, im_pred, im_err))

    # Animation update function
    def update(frame):
        for i in range(Nchannel):
            ims[i][0].set_array(gt_batch[i, :, :, frame])  # Update GT
            ims[i][1].set_array(pred_batch[i, :, :, frame])  # Update Prediction
            ims[i][2].set_array(error_batch[i, :, :, frame])  # Update Error
        return [im for triple in ims for im in triple]  # Flatten the list of images

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=100, blit=False)

    # Show animation
    plt.show()
