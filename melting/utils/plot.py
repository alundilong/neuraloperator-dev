import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Load the dataset
file_path = "../dataset/small/melting_train_50.pt"  # Change path if needed
data = torch.load(file_path)

# Extract the first sample
y_sample = data["y"][10]  # Shape: (num_channels, height, width, time_steps)

# Convert to numpy for plotting
y_sample_np = y_sample.numpy()  # Shape: (num_channels, height, width, time_steps)

# Extract dimensions
num_channels, height, width, time_steps = y_sample_np.shape
torch.set_printoptions(threshold=10_000_000, linewidth=200)
#np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
#print(data['x'].shape)
#print(y_sample[2,:50,:50,0])

# Dynamically determine rows and cols
num_cols = math.ceil(math.sqrt(num_channels))  # Try to make it a square grid
num_rows = math.ceil(num_channels / num_cols)

# Set up figure and subplots with dynamic layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))

# Flatten axes array if needed (when num_channels < num_rows * num_cols)
axes = np.array(axes).flatten()

# Create initial images and color bars
ims = []
cbars = []  # Store color bar
for i in range(num_channels):
    ax = axes[i]
    im = ax.imshow(y_sample_np[i, :, :, 0], cmap="viridis", animated=True)
    ax.set_title(f"C{i + 1}")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)  # Add colorbar
    ims.append(im)
    cbars.append(cbar)

# Hide any extra subplot axes (if num_channels < grid size)
for j in range(num_channels, len(axes)):
    axes[j].axis("off")  # Hide unused subplots

# Animation function
def update(frame):
    for i in range(num_channels):
        ims[i].set_array(y_sample_np[i, :, :, frame])  # Update each subplot with new frame
        vmin, vmax = y_sample_np[i, :, :, frame].min(), y_sample_np[i, :, :, frame].max()
        ims[i].set_clim(vmin, vmax)
        cbars[i].update_normal(ims[i])  # Update colorbar to match new limits
    return ims

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=500, blit=False)

# Adjust layout and show animation
plt.tight_layout()
plt.show()

