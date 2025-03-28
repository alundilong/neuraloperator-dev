import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import tempfile

from neuralop.data.transforms.minmax_normalizers import MinMaxNormalizer
from neuralop import get_model
from melting.utils.melting_formula import calculate_alpha
import torch

_model = None
_x_transformer = None
_y_transformer = None
_x_min_val = None
_x_max_val = None
_y_min_val = None
_y_max_val = None

def predict_melting(x_input):

    with torch.no_grad():
        x_input1 = _x_transformer.transform(x_input)
        out = _model(x_input1)
        out = _y_transformer.inverse_transform(out)

    return out

def sketch_to_mask(sketch_dict, nx, ny):
    if sketch_dict is None or "composite" not in sketch_dict:
        return np.zeros((nx, ny))
    sketch = np.array(sketch_dict["composite"])
    sketch = np.mean(sketch, axis=-1)
    sketch_resized = np.array(Image.fromarray(sketch).resize((ny, nx)))
    mask = (sketch_resized < 128).astype(float)
    return mask

def run_simulation(sketch, time_length, dt, nx, ny):
    # Given parameters
    box_range = (-0.5e-3, 99.5e-3)  # Corrected syntax (assuming 'a' was a typo)
    box_length = 100e-3
    mask = sketch_to_mask(sketch, nx, ny)  # Replace with your actual mask from sketch_to_mask()
    
    # Calculate derived parameters
    dx = dy = box_length / nx
    nt = int(np.floor(time_length / dt)) + 1
    
    # Create coordinate grids
    x = np.linspace(box_range[0] + dx/2, box_range[1] - dx/2, nx)  # Cell-centered coordinates
    y = np.linspace(box_range[0] + dy/2, box_range[1] - dy/2, ny)
    t = np.linspace(0, time_length, nt)
    
    # Create meshgrids
    X, Y = np.meshgrid(x, y, indexing='ij')  # Shape (nx, ny)
    T = np.linspace(0, time_length, nt)      # Shape (nt,)
    
    # Initialize the 5D tensor (batch, channel, nx, ny, nt)
    x_input = np.zeros((1, 8, nx, ny, nt)).astype(np.float32)
    y_output = np.zeros((1, 4, nx, ny, nt)).astype(np.float32)

    global _x_transformer, _y_transformer

    # Convert x_min_val to numpy if needed (if x_input is numpy)
    if isinstance(_x_min_val, torch.Tensor):
        x_min_val = _x_min_val.numpy()
    if isinstance(_x_max_val, torch.Tensor):
        x_max_val = _x_max_val.numpy()
    if isinstance(_y_min_val, torch.Tensor):
        y_min_val = _y_min_val.numpy()
    if isinstance(_y_max_val, torch.Tensor):
        y_max_val = _y_max_val.numpy()
    
    # Reshape and broadcast x_min_val to match x_input's channel dimension
    x_min_val = x_min_val.reshape(1, 8, 1, 1, 1) * np.ones_like(x_input)
    x_max_val = x_max_val.reshape(1, 8, 1, 1, 1) * np.ones_like(x_input)
    y_min_val = y_min_val.reshape(1, 4, 1, 1, 1) * np.ones_like(y_output)
    y_max_val = y_max_val.reshape(1, 4, 1, 1, 1) * np.ones_like(y_output)

    x_min_val = torch.from_numpy(x_min_val.astype(np.float32))
    x_max_val = torch.from_numpy(x_max_val.astype(np.float32))
    y_min_val = torch.from_numpy(y_min_val.astype(np.float32))
    y_max_val = torch.from_numpy(y_max_val.astype(np.float32))

    _x_transformer = MinMaxNormalizer(min_val=x_min_val,max_val=x_max_val)
    _y_transformer = MinMaxNormalizer(min_val=y_min_val,max_val=y_max_val)
    
    # Fill each channel
    x_input[0, 0, :, :, :] = X[:, :, np.newaxis]  # x coordinate (broadcast across nt)
    x_input[0, 1, :, :, :] = Y[:, :, np.newaxis]  # y coordinate (broadcast across nt)
    x_input[0, 2, :, :, :] = T[np.newaxis, np.newaxis, :]  # time (broadcast across nx, ny)
    
    # Constant fields
    x_input[0, 3, :, :, :] = 0.0        # p (pressure)
    x_input[0, 4, :, :, :] = 301.45     # T (temperature)
    x_input[0, 5, :, :, :] = mask[:, :, np.newaxis]  # mask (broadcast across nt)
    x_input[0, 6, :, :, :] = 0.0        # Ux (x-velocity)
    x_input[0, 7, :, :, :] = 0.0        # Uy (y-velocity)
    x_input = torch.from_numpy(x_input.astype(np.float32))

    prediction = predict_melting(x_input)
    mask = x_input[:,5:6,...]
    frames = mask*calculate_alpha(prediction[:,1:2,...],T_l=303.43,T_s=302.43)
    frames = frames[0,0]

    fig, ax = plt.subplots()
    im = ax.imshow(frames[:, :, 0], cmap="plasma", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("Melting Progress (t=0)")

    def update(t):
        current_frame = frames[:, :, t].cpu().numpy()
        
        # Update image data and normalization
        im.set_array(current_frame)
        im.set_clim(vmin=np.min(current_frame), vmax=np.max(current_frame))
        
        # Update colorbar
        cbar.update_normal(im)
        
        # Update title with current min/max values
        ax.set_title(f"Time Step {t}\nMin: {np.min(current_frame):.2f}, Max: {np.max(current_frame):.2f}")
        
        return [im]

    ani = FuncAnimation(fig, update, frames=frames.shape[-1], blit=True)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        ani.save(f.name, writer="pillow", fps=5)
        gif_path = f.name

    plt.close(fig)
    return gif_path

def init_canvas():
    white = np.ones((200, 200, 3), dtype=np.uint8) * 255
    return {
        "background": white,
        "layers": [],
        "composite": white.copy()
    }


def create_app(config=None):

    global _model, _x_transformer, _y_transformer
    global _x_min_val, _x_max_val, _y_min_val, _y_max_val

    if config is None:
        # throw error
        raise ValueError("Config cannot be None. Please provide a valid configuration.")
    else:
        _model = get_model(config)
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model.load_checkpoint(save_folder=config.tfno3d.save_dir, save_name="model", map_location=map_location)

        _x_min_val = torch.tensor(config.eval.x_min)
        _x_max_val = torch.tensor(config.eval.x_max)

        _y_min_val = torch.tensor(config.eval.y_min)
        _y_max_val = torch.tensor(config.eval.y_max)

    with gr.Blocks(title="Melting Simulation") as demo:
        gr.Markdown("## 🧊 Draw Porous Structure and Simulate Melting")
    
        with gr.Row():
            with gr.Column():
                nx_slider = gr.Slider(label="Grid Width (nx)", minimum=10, maximum=100, value=50, step=1)
                ny_slider = gr.Slider(label="Grid Height (ny)", minimum=10, maximum=100, value=50, step=1)
                time_length_slider = gr.Slider(label="Total Time (s)", minimum=1, maximum=4000, value=2000, step=20)
                dt_slider = gr.Slider(label="Time Resolution (dt)", minimum=20, maximum=400, value=200, step=1)
                run_btn = gr.Button("Run Simulation", variant="primary")
    
            with gr.Column():
                canvas = gr.Sketchpad(
                    label="Draw porous regions (black = porous)",
                    brush=10,
                    height=300,
                    width=300,
                    value=init_canvas()
                )
                output_anim = gr.Image(label="Melting Animation", type="filepath")
    
        run_btn.click(
            fn=run_simulation,
            inputs=[canvas, time_length_slider, dt_slider, nx_slider, ny_slider],
            outputs=output_anim
        )
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch()
