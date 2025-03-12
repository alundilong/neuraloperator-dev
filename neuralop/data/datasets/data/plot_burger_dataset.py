import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

if __name__ == "__main__":

    # Load data
    fn = "tmp/burgers_train_16.pt"  # Replace with your actual file

    # Load data
    data = torch.load(fn)
    
    # Extract ground truth
    y = data['y']  # Shape: (B, C, T, D)
    
    id0 = 0
    id1 = -1
    # Select first two batches (B=0 and B=1) and first channel (C=0)
    y_selected_0 = y[id0, 0, :, :].cpu().numpy()  # Shape: (T, D)
    y_selected_1 = y[id1, 0, :, :].cpu().numpy()  # Shape: (T, D)
    
    # Create figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    lines = [axs[0].plot([], [], lw=2)[0], axs[1].plot([], [], lw=2)[0]]
    
    # Set plot limits
    for ax in axs:
        ax.set_xlim(0, y.shape[3])  # Spatial dimension
        ax.set_ylim(y.min(), y.max())  # Value range
        ax.set_xlabel("Spatial Position")
        ax.set_ylabel("Value")
    
    axs[0].set_title("Spatial Variation Over Time (Batch 0, Channel 0)")
    axs[1].set_title("Spatial Variation Over Time (Batch 1, Channel 0)")
    
    # Initialization function
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    
    # Animation function
    def update(frame):
        lines[0].set_data(np.arange(y.shape[3]), y_selected_0[frame])  # Batch 0
        lines[1].set_data(np.arange(y.shape[3]), y_selected_1[frame])  # Batch 1
        axs[0].set_title(f"Batch 0 - Time Step: {frame}")
        axs[1].set_title(f"Batch 1 - Time Step: {frame}")
        return lines
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=y.shape[2], init_func=init, blit=True, interval=100)
    
    # Show animation
    plt.show()
