import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
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
    sketch_resized = np.array(Image.fromarray(sketch).resize((ny, nx)))
    mask = (sketch_resized < 128).astype(float)
    return mask

def run_simulation(sketch, time_length, dt, nx, ny):
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

def init_canvas():
    white = np.ones((200, 200, 3), dtype=np.uint8) * 255
    return {
        "background": white,
        "layers": [],
        "composite": white.copy()
    }


def create_app():
    with gr.Blocks(title="Melting Simulation") as demo:
        gr.Markdown("## 🧊 Draw Porous Structure and Simulate Melting")
    
        with gr.Row():
            with gr.Column():
                nx_slider = gr.Slider(label="Grid Width (nx)", minimum=10, maximum=100, value=50, step=1)
                ny_slider = gr.Slider(label="Grid Height (ny)", minimum=10, maximum=100, value=50, step=1)
                time_length_slider = gr.Slider(label="Total Time (s)", minimum=1, maximum=100, value=10, step=1)
                dt_slider = gr.Slider(label="Time Resolution (dt)", minimum=0.1, maximum=1.0, value=0.2, step=0.1)
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
