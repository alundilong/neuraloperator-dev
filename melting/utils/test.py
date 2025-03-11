import unittest
import torch

class TestMeltingDataset(unittest.TestCase):

    def check_melting_data(self, file_path, expected_x_shape, expected_y_shape):
        """Helper function to verify x and y shapes inside the .pt file"""
        try:
            # Load the .pt file
            data = torch.load(file_path)

            for key, value in data.items():
                print(f"{key}: shape {value.shape}, dtype {value.dtype}, size {value.numel() * value.element_size() / (1024**2):.2f} MB")

            # Extract x and y tensors
            x, y = data["x"], data["y"]

            # Test if shapes match
            self.assertEqual(x.shape, expected_x_shape, f"x shape mismatch: expected {expected_x_shape}, got {x.shape}")
            self.assertEqual(y.shape, expected_y_shape, f"y shape mismatch: expected {expected_y_shape}, got {y.shape}")

            print(f"{file_path}: Shapes are correct! (x: {x.shape}, y: {y.shape})")

        except Exception as e:
            self.fail(f"Failed to load {file_path}: {e}")

    def test_melting_train(self):
        """Test the training dataset shape"""
        num_snapshots = 1500
        train_split = 0.80
        train_size = int(train_split * num_snapshots)
        expected_x_shape = (train_size, 9, 50, 50, 11)  # Example shape
        expected_y_shape = (train_size, 5, 50, 50, 11)
        self.check_melting_data("../dataset/melting_train_50.pt", expected_x_shape, expected_y_shape)

    def test_melting_test(self):
        """Test the test dataset shape"""
        num_snapshots = 1500
        train_split = 0.80
        train_size = int(train_split * num_snapshots)
        test_size = num_snapshots - train_size
        expected_x_shape = (test_size, 9, 50, 50, 11)  # Example shape
        expected_y_shape = (test_size, 5, 50, 50, 11)
        self.check_melting_data("../dataset/melting_test_50.pt", expected_x_shape, expected_y_shape)

if __name__ == "__main__":
    unittest.main()

