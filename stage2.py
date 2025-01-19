from utils import train, logger, FTConfig
import numpy as np
import torch
import torch.nn.functional as F
import random

def get_samples(
    file_path: str = "tmp/cifar10_train.npy", 
    sample_ratio: float = 0.02, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = 'tmp',
    temperature: float = 0.07,
    iterations: int = 300,
    balance: float = 1.0,
    slice: int = None,
    batch_size: int = 100000,
):
    r"""
        Get feratures from the file, and then train the model to get the centroid samples
        
        Args:
            file_path ('str', defaults to 'tmp/cifar10_train.npy'):
                the path of the file
            sample_ratio ('float', defaults to 0.02):
                the ratio of the samples
            device ('str', defaults to 'cuda' if torch.cuda.is_available() else 'cpu'):
                the device to run the model
            save_dir ('str', defaults to 'tmp'):
                the directory to save the centroid samples
            temperature ('float', defaults to 0.07):
                the temperature of the scaled similarity
            iterations ('int', defaults to 100):
                the number of iterations
            balance ('float', defaults to 1.0):
                the balance of tge loss
            slice ('int', defaults to None):
                the slice of the samples
            batch_size ('int', defaults to 100000):
                the number of samples used in each iteration
        
        Returns:
            the centroid samples
    """
    inp = np.load(file_path)
    features, _ = inp[:, :-1], inp[:, -1]
    features = torch.tensor(features, dtype=torch.float32).to(device)
    features = F.normalize(features, dim=-1)
    
    total_num = features.size(0)
    sample_num = int(total_num * sample_ratio)
    
    train(
        features=features,
        sample_num=sample_num,
        temperature=temperature,
        balance=balance,
        slice=slice,
        device=device,
        save_dir=save_dir,
        iterations=iterations,
        batch_size=batch_size,
    )
    
    
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    get_samples()