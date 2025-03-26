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

def compare_tensors_animation(pred: torch.Tensor, gt: torch.Tensor, batch_num: int = 0, sample_num: int = 0):
    """
    Creates an animation comparing predicted and ground truth tensors with dynamic colorbar updates.

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

    title = {0:"p",1:"T",2:"Ux",3:"Uy"}
    if Nchannel == 1:
        axes = axes.reshape(1, -1)
        title = {0:"T"}

    ims = []
    colorbars = []
    for i in range(Nchannel):
        # Initialize with the first time step
        vmin, vmax = np.min(gt_batch[i, :, :, 0]), np.max(gt_batch[i, :, :, 0])
        vmin_err, vmax_err = np.min(error_batch[i, :, :, 0]), np.max(error_batch[i, :, :, 0])

        # Plot Ground Truth
        ax = axes[i, 0]
        im_gt = ax.imshow(gt_batch[i, :, :, 0], cmap="viridis", animated=True, vmin=vmin, vmax=vmax)
        ax.set_title(f"GT-{title[i]}")
        cbar_gt = fig.colorbar(im_gt, ax=ax, orientation="vertical")
        colorbars.append(cbar_gt)

        # Plot Prediction (Using same vmin/vmax as GT)
        ax = axes[i, 1]
        im_pred = ax.imshow(pred_batch[i, :, :, 0], cmap="viridis", animated=True, vmin=vmin, vmax=vmax)
        ax.set_title(f"Pred-{title[i]}")
        cbar_pred = fig.colorbar(im_pred, ax=ax, orientation="vertical")
        colorbars.append(cbar_pred)

        # Plot Error (Independent color scale)
        ax = axes[i, 2]
        im_err = ax.imshow(error_batch[i, :, :, 0], cmap="inferno", animated=True, vmin=vmin_err, vmax=vmax_err)
        ax.set_title(f"Error-{title[i]}")
        cbar_err = fig.colorbar(im_err, ax=ax, orientation="vertical")
        colorbars.append(cbar_err)

        ims.append((im_gt, im_pred, im_err))

    # Animation update function
    def update(frame):
        for i in range(Nchannel):
            # Update vmin/vmax dynamically based on the current frame
            vmin, vmax = np.min(gt_batch[i, :, :, frame]), np.max(gt_batch[i, :, :, frame])
            vmin_err, vmax_err = np.min(error_batch[i, :, :, frame]), np.max(error_batch[i, :, :, frame])

            ims[i][0].set_array(gt_batch[i, :, :, frame])  # Update GT
            ims[i][1].set_array(pred_batch[i, :, :, frame])  # Update Prediction
            ims[i][2].set_array(error_batch[i, :, :, frame])  # Update Error

            # Update color scales for Ground Truth and Prediction (same scale)
            ims[i][0].set_clim(vmin, vmax)
            ims[i][1].set_clim(vmin, vmax)

            # Update color scale for Error independently
            ims[i][2].set_clim(vmin_err, vmax_err)

            # Update colorbar limits
            colorbars[i * 3].mappable.set_clim(vmin, vmax)  # GT colorbar
            colorbars[i * 3 + 1].mappable.set_clim(vmin, vmax)  # Pred colorbar
            colorbars[i * 3 + 2].mappable.set_clim(vmin_err, vmax_err)  # Error colorbar

        return [im for triple in ims for im in triple]  # Flatten the list of images

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=100, blit=False)
    ani.save(f"animation_s{sample_num}_b{batch_num}.gif", writer=animation.PillowWriter(fps=20))
    # Show animation
    plt.show()

def compare_batches_animation(pred: torch.Tensor, 
                            gt: torch.Tensor, 
                            sample_num: int = 0,
                            channel_names: dict = {0: "T"}):
    """
    Creates an animation comparing multiple batches side by side, with each row showing one batch's GT, Pred, and Error.
    
    Parameters:
    - pred (torch.Tensor): Predicted tensor of shape (Nbatch, Nchannel, Nx, Ny, Nt)
    - gt (torch.Tensor): Ground truth tensor (same shape as pred)
    - batch_nums (list): List of batch indices to visualize
    - channel_names (dict): Dictionary mapping channel indices to names
    
    Returns:
    - None (Displays and saves an animation)
    """
    # Convert to NumPy if tensors are PyTorch tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    # Validate shapes
    if pred.shape != gt.shape:
        raise ValueError("Predicted and ground truth tensors must have the same shape.")

    Nbatch, Nchannel, Nx, Ny, Nt = pred.shape
    
    batch_nums = np.arange(Nbatch)

    # Determine subplot layout (3 columns: GT, Pred, Error)
    nrows = len(batch_nums)
    ncols = 3

    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)  # Ensure axes is 2D even for single batch

    ims = []
    colorbars = []

    # Initialize all plots
    for row, batch_num in enumerate(batch_nums):
        pred_batch = pred[batch_num]  # (Nchannel, Nx, Ny, Nt)
        gt_batch = gt[batch_num]      # (Nchannel, Nx, Ny, Nt)
        error_batch = np.abs(pred_batch - gt_batch)

        # For each channel, we'll show the mean across channels (or you could modify to show specific channel)
        # Here I'm showing the mean, but you could modify to show a specific channel
        pred_mean = np.mean(pred_batch, axis=0)  # (Nx, Ny, Nt)
        gt_mean = np.mean(gt_batch, axis=0)      # (Nx, Ny, Nt)
        error_mean = np.mean(error_batch, axis=0) # (Nx, Ny, Nt)

        # Get initial min/max values
        vmin, vmax = np.min(gt_mean[:, :, 0]), np.max(gt_mean[:, :, 0])
        vmin_err, vmax_err = np.min(error_mean[:, :, 0]), np.max(error_mean[:, :, 0])

        # Plot Ground Truth
        ax = axes[row, 0]
        im_gt = ax.imshow(gt_mean[:, :, 0], cmap="viridis", animated=True, vmin=vmin, vmax=vmax)
        ax.set_title(f"Batch {batch_num} - GT (mean)")
        cbar_gt = fig.colorbar(im_gt, ax=ax)
        colorbars.append(cbar_gt)

        # Plot Prediction
        ax = axes[row, 1]
        im_pred = ax.imshow(pred_mean[:, :, 0], cmap="viridis", animated=True, vmin=vmin, vmax=vmax)
        ax.set_title(f"Batch {batch_num} - Pred (mean)")
        cbar_pred = fig.colorbar(im_pred, ax=ax)
        colorbars.append(cbar_pred)

        # Plot Error
        ax = axes[row, 2]
        im_err = ax.imshow(error_mean[:, :, 0], cmap="inferno", animated=True, vmin=vmin_err, vmax=vmax_err)
        ax.set_title(f"Batch {batch_num} - Error (mean)")
        cbar_err = fig.colorbar(im_err, ax=ax)
        colorbars.append(cbar_err)

        ims.append((im_gt, im_pred, im_err, gt_mean, pred_mean, error_mean))

    # Animation update function
    def update(frame):
        for row in range(nrows):
            gt_data = ims[row][3][:, :, frame]
            pred_data = ims[row][4][:, :, frame]
            err_data = ims[row][5][:, :, frame]

            # Update data
            ims[row][0].set_array(gt_data)
            ims[row][1].set_array(pred_data)
            ims[row][2].set_array(err_data)

            # Update color limits
            vmin, vmax = np.min(gt_data), np.max(gt_data)
            vmin_err, vmax_err = np.min(err_data), np.max(err_data)

            ims[row][0].set_clim(vmin, vmax)
            ims[row][1].set_clim(vmin, vmax)
            ims[row][2].set_clim(vmin_err, vmax_err)

            # Update colorbars
            colorbars[row * 3].mappable.set_clim(vmin, vmax)
            colorbars[row * 3 + 1].mappable.set_clim(vmin, vmax)
            colorbars[row * 3 + 2].mappable.set_clim(vmin_err, vmax_err)

        return [im for triple in ims for im in triple[:3]]  # Flatten the list

    # Create and save animation
    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=100, blit=False)
    fieldname = channel_names[0]
    ani.save(f"batch_comparison_s{sample_num}_{fieldname}.gif", writer=animation.PillowWriter(fps=20))
    plt.show()
    #plt.close()
    return ani
