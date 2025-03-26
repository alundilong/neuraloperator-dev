import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import json

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

# Convert drawing to mask
def update_mask_from_drawings(drawing_data, nx, ny):
    mask = np.zeros((ny, nx))  # shape (rows, cols) = (y, x)

    for shape in drawing_data.get("shapes", []):
        if shape["shape_type"] == "rect":
            x0, y0 = int(shape["x0"] * nx), int(shape["y0"] * ny)
            x1, y1 = int(shape["x1"] * nx), int(shape["y1"] * ny)
            mask[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)] = 1

        elif shape["shape_type"] == "circle":
            cx, cy = int(shape["cx"] * nx), int(shape["cy"] * ny)
            r = int(shape["r"] * max(nx, ny))
            y, x = np.ogrid[:ny, :nx]
            mask[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = 1

    return mask

# Create Gradio app
with gr.Blocks(title="Melting Simulation") as demo:
    gr.Markdown("""
    # =% Melting Process Simulator
    Draw pores on the canvas below (rectangles/circles)
    """)

    with gr.Row():
        with gr.Column():
            nx = gr.Slider(label="Grid Width (nx)", minimum=5, maximum=50, value=10, step=1)
            ny = gr.Slider(label="Grid Height (ny)", minimum=5, maximum=50, value=10, step=1)

            drawing_mode = gr.Radio(
                ["rectangle", "circle", "freehand"],
                label="Drawing Tool",
                value="rectangle"
            )
            clear_btn = gr.Button("Reset Canvas")

            time_length = gr.Slider("Total Time (s)", 1, 100, 10, step=1)
            dt = gr.Slider("Time Resolution (dt)", 0.1, 1.0, 0.2, step=0.1)
            run_btn = gr.Button("Run Simulation", variant="primary")

            drawing_data = gr.JSON(value={"shapes": []}, visible=False)

        with gr.Column():
            drawing_canvas = gr.ImageEditor(
                interactive=True,
                transforms=[],
                image_mode="L",
                width=400,
                height=400
            )
            output_plot = gr.Plot(label="Melting at Final Time Step")

    def init_canvas(nx_val, ny_val):
        return np.ones((ny_val, nx_val), dtype=np.uint8) * 255

    nx.change(init_canvas, [nx, ny], [drawing_canvas])
    ny.change(init_canvas, [nx, ny], [drawing_canvas])

    def clear_canvas(nx_val, ny_val):
        return init_canvas(nx_val, ny_val), {"shapes": []}

    clear_btn.click(
        clear_canvas,
        inputs=[nx, ny],
        outputs=[drawing_canvas, drawing_data]
    )

    def run_simulation(drawing_json, nx_val, ny_val, time_length_val, dt_val):
        dx, dy = 0.1, 0.1
        mask = update_mask_from_drawings(drawing_json, nx_val, ny_val)
        prediction = predict_melting(mask, time_length_val, dx, dy, dt_val)
        final_frame = prediction[0, 0, :, :, -1]

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(final_frame, cmap="plasma", vmin=0, vmax=1)
        ax.set_title("Melting Progress at Final Time")
        plt.colorbar(im, ax=ax, label="Melting")
        return fig

    run_btn.click(
        run_simulation,
        inputs=[drawing_data, nx, ny, time_length, dt],
        outputs=output_plot
    )

demo.launch()

