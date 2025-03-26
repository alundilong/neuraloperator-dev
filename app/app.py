import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile

def predict_melting(mask, time_length, dx, dy, dt):
    nx, ny = mask.shape
    nt = int(time_length / dt)
    simulation = np.zeros((1, 1, nx, ny, nt))
    for t in range(nt):
        simulation[0, 0, :, :, t] = (
            np.linspace(0, 1, nx).reshape(-1, 1) * 0.5 +
            np.random.normal(0, 0.1, (nx, ny)) * (1 + mask * 0.5)
        )
        simulation[0, 0, :, :, t] = np.clip(simulation[0, 0, :, :, t], 0, 1) * (t / nt)
    return simulation

def sketch_to_mask(sketch_dict):
    if sketch_dict is None or "composite" not in sketch_dict:
        return np.zeros((50, 50))
    sketch = np.array(sketch_dict["composite"])  # Extract actual image
    sketch = np.mean(sketch, axis=-1)  # Convert to grayscale
    mask = (sketch < 128).astype(float)
    return mask

def run_simulation(sketch, time_length, dt):
    dx = dy = 0.1
    mask = sketch_to_mask(sketch)
    prediction = predict_melting(mask, time_length, dx, dy, dt)
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

with gr.Blocks(title="Melting Simulation") as demo:
    gr.Markdown("## 🧊 Draw Porous Structure (Freehand Drawing)")

    with gr.Row():
        with gr.Column():
            nx = gr.Slider(label="Grid Width (nx)", minimum=5, maximum=100, value=50, step=1)
            ny = gr.Slider(label="Grid Height (ny)", minimum=5, maximum=100, value=50, step=1)

            time_length = gr.Slider("Total Time (s)", 1, 100, 10, step=1)
            dt = gr.Slider("Time Resolution (dt)", 0.1, 1.0, 0.2, step=0.1)

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

    def reset_canvas():
        return np.ones((50, 50, 3), dtype=np.uint8) * 255  # White

    clear_btn.click(fn=reset_canvas, inputs=[], outputs=canvas)
    run_btn.click(fn=run_simulation, inputs=[canvas, time_length, dt], outputs=output_anim)

demo.launch()
