import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Print the CUDA device count and name(s)
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    # Print CUDA version
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA is not available.")
