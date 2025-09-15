import torch

def print_vram_usage():
    device_index = 0  # change if you have multiple GPUs
    allocated_memory = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
    print(f"Allocated memory: {allocated_memory:.2f} MB")
    print(f"Reserved memory:  {reserved_memory:.2f} MB")

def get_string_vram_usage() -> str:
    device_index = 0  # change if you have multiple GPUs
    allocated_memory = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
    return f"Allocated memory: {allocated_memory:.2f} MB, Reserved memory: {reserved_memory:.2f} MB"

if __name__ == "__main__":
    print_vram_usage()
    # Example usage
    vram_usage = get_string_vram_usage()
    print(vram_usage)
