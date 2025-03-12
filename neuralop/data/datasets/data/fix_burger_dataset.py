import torch

def fix_file(fn):
    fn2 = f"olddata/{fn}"
    data = torch.load(fn2)
    print(data['x'].shape)
    print(data['y'].shape)
    # Reshape data['x']
    data['x'] = data['x'].unsqueeze(1).unsqueeze(2).expand(-1, -1, 17, -1)  # Shape: (800, 1, 17, 16)
    # Reshape data['y']
    data['y'] = data['y'].unsqueeze(1)  # Shape: (800, 1, 17, 16)
    # Print new shapes
    print(data['x'].shape)
    print(data['y'].shape)
    
    torch.save(data,f'tmp/{fn}')

if __name__ == "__main__":
    fn ="burgers_train_16.pt"
    fix_file(fn)
    fn ="burgers_test_16.pt"
    fix_file(fn)
