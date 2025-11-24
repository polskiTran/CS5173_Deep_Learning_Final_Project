import GPUtil
import torch

def check_gpu():    
    print("(*) Get GPU info")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}, Load: {gpu.load*100}%, Free Memory: {gpu.memoryFree}MB, Used Memory: {gpu.memoryUsed}MB, Total Memory: {gpu.memoryTotal}MB, Temperature: {gpu.temperature}°C")
    print("\n(*) Check if PyTorch can access GPU")
    if torch.cuda.is_available():
        print(f"PyTorch can access GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch cannot access GPU")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return device


def main():
    print("Hello from cs5173-deep-learning-final-project!")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
    
    device = check_gpu()
    main()
