import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile

def predict_melting(mask, time_length, dx, dy, dt, nx, ny):
    nt = int(time_length / dt)
    simulation = np.zeros((1, 1, nx, ny, nt))
    for t in range(nt):
        simulation[0, 0, :, :, t] = (
            np.linspace(0, 1, nx).reshape(-1, 1) * 0.5 +
            np.random.normal(0, 0.1, (nx, ny)) * (1 + mask * 0.5)
        )
        simulation[0, 0, :, :, t] = np.clip(simulation[0, 0, :, :, t], 0, 1) * (t / nt)
    return simulation

def sketch_to_mask(sketch_dict, nx, ny):
    if sketch_dict is None or "composite" not in sketch_dict:
        return np.zeros((nx, ny))
    sketch = np.array(sketch_dict["composite"])
    sketch = np.mean(sketch, axis=-1)
    # Resize to simulation resolution if needed
    sketch_resized = np.array(Image.fromarray(sketch).resize((ny, nx)))  # (width, height)
    mask = (sketch_resized < 128).astype(float)
    return mask

def run_simulation(sketch, time_length, dt, nx, ny):
    if time_length is None:
        time_length = 10
    if dt is None or dt == 0:
        dt = 0.2
    if nx is None:
        nx = 50
    if ny is None:
        ny = 50

    dx = dy = 0.1
    mask = sketch_to_mask(sketch, nx, ny)
    prediction = predict_melting(mask, time_length, dx, dy, dt, nx, ny)
    frames = prediction[0, 0]

    fig, ax = plt.subplots()
    im = ax.imshow(frames[:, :, 0], cmap="plasma", vmin=0, vmax=1)
    ax.set_title("Melting Progress")

    def update(t):
        im.set_array(frames[:, :, t])
        ax.set_title(f"Time Step {t}")
        return [im]

    ani = FuncAnimation(fig, update, frames=frames.shape[-1], blit=True)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        ani.save(f.name, writer="pillow", fps=5)
        gif_path = f.name

    plt.close(fig)
    return gif_path

# Import PIL for resizing
from PIL import Image

with gr.Blocks(title="Melting Simulation") as demo:
    gr.Markdown("## 🧊 Draw Porous Structure (Freehand Drawing)")

    with gr.Row():
        with gr.Column():
            nx_slider = gr.Slider(label="Grid Width (nx)", minimum=5, maximum=100, value=50, step=1)
            ny_slider = gr.Slider(label="Grid Height (ny)", minimum=5, maximum=100, value=50, step=1)

            time_length_slider = gr.Slider("Total Time (s)", 1, 100, 10, step=1)
            dt_slider = gr.Slider("Time Resolution (dt)", 0.1, 1.0, 0.2, step=0.1)

            run_btn = gr.Button("Run Simulation", variant="primary")
            clear_btn = gr.Button("Clear Drawing")

        with gr.Column():
            canvas = gr.Sketchpad(
                label="Draw porous regions (black = porous)",
                brush=10,
                height=250,
                width=250
            )
            output_anim = gr.Image(label="Melting Animation", type="filepath")

    def reset_canvas(nx, ny):
        return np.ones((ny, nx, 3), dtype=np.uint8) * 255  # Note: shape is (height, width, channels)

    clear_btn.click(fn=reset_canvas, inputs=[nx_slider, ny_slider], outputs=canvas)

    run_btn.click(
        fn=run_simulation,
        inputs=[canvas, time_length_slider, dt_slider, nx_slider, ny_slider],
        outputs=output_anim
    )

demo.launch()
