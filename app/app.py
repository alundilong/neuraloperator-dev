import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Fake model prediction
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

# Convert sketch to binary mask
def sketch_to_mask(sketch):
    if sketch is None:
        return np.zeros((50, 50))
    # Assume black drawing = porous
    sketch = np.mean(sketch, axis=-1)  # Convert to grayscale
    mask = (sketch < 128).astype(float)
    return mask

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
                height=350,
                width=350
            )
            output_plot = gr.Plot(label="Melting at Final Time Step")

    def reset_canvas():
        return np.ones((50, 50, 3), dtype=np.uint8) * 255  # White image

    clear_btn.click(fn=reset_canvas, inputs=[], outputs=canvas)

    def run_simulation(sketch, time_length, dt):
        dx = dy = 0.1
        mask = sketch_to_mask(sketch)
        prediction = predict_melting(mask, time_length, dx, dy, dt)
        final_frame = prediction[0, 0, :, :, -1]

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(final_frame, cmap="plasma", vmin=0, vmax=1)
        ax.set_title("Melting Progress at Final Time")
        plt.colorbar(im, ax=ax)
        return fig

    run_btn.click(fn=run_simulation, inputs=[canvas, time_length, dt], outputs=output_plot)

demo.launch()
