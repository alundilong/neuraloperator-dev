import numpy as np
import torch
import pandas as pd
import argparse
import os

def process_file(input_path, output_path, save_dir, train_split=0.8):
    # Load input data (xx.dat)
    data_x = pd.read_csv(input_path, delimiter=',', header=None).values

    # Ensure the number of rows is a multiple of (2500 * 11)
    num_rows_per_snapshot = 2500 * 11
    num_snapshots = data_x.shape[0] // num_rows_per_snapshot
    valid_rows = num_snapshots * num_rows_per_snapshot

    if data_x.shape[0] % num_rows_per_snapshot != 0:
        print(f"Warning: {data_x.shape[0] - valid_rows} rows removed from {input_path}.")
        data_x = data_x[:valid_rows]

    # Extract X, Y, T (first 3 columns) and values (remaining columns)
    x, y, t = data_x[:, 0:1], data_x[:, 1:2], data_x[:, 2:3]  # First 3 columns
    values = data_x[:, 3:]  # Remaining columns are data channels

    # Combine x, y, t as additional channels
    values_with_coordinates = np.concatenate([x, y, t, values], axis=1)
    k = values_with_coordinates.shape[1]  # Number of channels (x, y, t + other values)

    # Define spatial dimensions
    d0, d1, d2 = 50, 50, 11

    # Reshape input data to (batch, k, d0, d1, d2)
    data_x_reshaped = values_with_coordinates.reshape(num_snapshots, d0, d1, d2, k).transpose(0, 4, 1, 2, 3)

    # Convert to PyTorch tensor
    tensor_x = torch.tensor(data_x_reshaped, dtype=torch.float32)

    # Load output data (y.dat)
    data_y = pd.read_csv(output_path, delimiter=',', header=None).values

    # Ensure output matches input structure
    if data_y.shape[0] != valid_rows:
        raise ValueError(f"Error: Output file {output_path} does not match input file {input_path} in row count.")

    # Extract number of output channels
    k_out = data_y.shape[1]

    # Reshape output data to (batch, k_out, d0, d1, d2)
    data_y_reshaped = data_y.reshape(num_snapshots, d0, d1, d2, k_out).transpose(0, 4, 1, 2, 3)

    # Convert to PyTorch tensor
    tensor_y = torch.tensor(data_y_reshaped, dtype=torch.float32)

    # Shuffle dataset (shuffle indices in batch dimension)
    indices = torch.randperm(num_snapshots)  # Generates a shuffled list of indices
    tensor_x = tensor_x[indices]
    tensor_y = tensor_y[indices]

    # Split into train/test sets (80% train, 20% test)
    train_size = int(train_split * num_snapshots)
    train_x, test_x = tensor_x[:train_size], tensor_x[train_size:]
    train_y, test_y = tensor_y[:train_size], tensor_y[train_size:]

    # Save train set
    train_save_path = os.path.join(save_dir, "melting_train_50.pt")
    torch.save({"x": train_x, "y": train_y}, train_save_path)

    # Save test set
    test_save_path = os.path.join(save_dir, "melting_test_50.pt")
    torch.save({"x": test_x, "y": test_y}, test_save_path)

    print(f"Train set saved to: {train_save_path}, shape: {train_x.shape}, {train_y.shape}")
    print(f"Test set saved to: {test_save_path}, shape: {test_x.shape}, {test_y.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process x.dat and y.dat into shuffled train/test PyTorch dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input x.dat file")
    parser.add_argument("--output", type=str, required=True, help="Path to output y.dat file")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save processed data")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training (default: 0.8)")

    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Process the files
    process_file(args.input, args.output, args.save_dir, args.train_split)
