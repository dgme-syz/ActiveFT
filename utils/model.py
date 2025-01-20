import torch
from torch import nn
from typing import List
import torch.nn.functional as F
import random
import wandb
from tqdm import tqdm
from .log import logger
from .config import FTConfig
import math

class SampleModel(nn.Module):
    def __init__(
        self,
        features: torch.Tensor,
        **kwargs
    ):
        r"""
            Args:
                features ('torch.Tensor'):
                    the output of the model
            Returns:
                the output of the model
        """
        super(SampleModel, self).__init__()
        self.features = features
        self.config = FTConfig(**kwargs)
        self.sample_num = self.config.sample_num
        self.temperature = self.config.temperature
        self.balance = self.config.balance
        self.total_num = features.size(0)
        self.slice = self.config.slice
        if self.slice == None:
            self.slice = self.total_num
        self.batch_size = self.config.batch_size
        print(self.config)
        self.centroids = nn.Parameter(self.init_centroids()).to(self.config.device)

    def init_centroids(self):
        r"""
            Initialize the centroids
        """
        sample_ids = list(range(self.total_num))
        sample_ids = random.sample(sample_ids, self.sample_num)
        
        return self.features[sample_ids].clone()

    def compute_loss(self):
        centroids = F.normalize(self.centroids, dim=-1)
        # Too many samples, we need to get some of them
        if self.batch_size > self.total_num:
            self.batch_size = self.total_num
            sample_ids = list(range(self.total_num))
        else:
            sample_ids = random.sample(list(range(self.total_num)), self.batch_size)
        features = self.features[sample_ids]
        sample_slice_num = math.ceil(1.0 * self.sample_num / self.slice)
        batch_slice_num = math.ceil(1.0 * self.batch_size / self.slice)

        prod_exp_pos = []
        pos_k = []
        for sid in range(batch_slice_num):
            start = sid * self.slice
            end = min((sid + 1) * self.slice, self.batch_size)
            prod = torch.matmul(features[start: end], centroids.transpose(1, 0))  # (slice_num, k)
            prod = prod / self.temperature
            prod_exp = torch.exp(prod)
            prod_exp_pos_cur, pos_k_cur = torch.max(prod_exp, dim=1)  # (slice_num, )
            prod_exp_pos.append(prod_exp_pos_cur)
            pos_k.append(pos_k_cur)
        pos_k = torch.cat(pos_k, dim=0)
        prod_exp_pos = torch.cat(prod_exp_pos, dim=0)

        cent_prob_exp_sum = []
        for sid in range(sample_slice_num):
            start = sid * self.slice
            end = min((sid + 1) * self.slice, self.sample_num)
            cent_prod = torch.matmul(centroids.detach(), centroids[start:end].transpose(1, 0))  # (k, slice_num)
            cent_prod = cent_prod / self.temperature
            cent_prod_exp = torch.exp(cent_prod)
            cent_prob_exp_sum_cur = torch.sum(cent_prod_exp, dim=0)  # (slice_num, )
            cent_prob_exp_sum.append(cent_prob_exp_sum_cur)
        cent_prob_exp_sum = torch.cat(cent_prob_exp_sum, dim=0)

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)

        return J

def train(
    features: torch.Tensor,
    sample_num: int, 
    temperature: float = 0.07,
    balance: float = 1.0,
    slice: int | None = None,
    batch_size: int = 100000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    iterations: int = 300,
    save_dir: str = 'tmp',
    scheduler_type: str | None = "cosine",
) -> List[int]:
    r"""
        Train the model to get the samples
        
        Args:
            features ('torch.Tensor'):
                the output of the model
            sample_num ('int'):
                the number of samples
            temperature ('float'):
                the temperature of the softmax
            balance ('float'):
                the balance of the loss
            slice ('int'):
                the slice of the samples
            batch_size ('int'):
                the batch size
            device ('str'):
                the device to use
            iterations ('int'):
                the number of iterations
            save_dir ('str'):
                the directory to save the config file
            scheduler_type ('str'):
                the type of the scheduler
                
        Returns:
            the sampled ids
    """
    
    model = SampleModel(
        features=features,
        sample_num=sample_num,
        temperature=temperature,
        balance=balance,
        slice=slice,
        batch_size=batch_size,
        device=device,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if scheduler_type != None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=1e-6)
    run = wandb.init(project="activate learning", config=model.config)
    logger.info(f"Start training the model for {iterations} iterations")
    with tqdm(range(iterations)) as pbar:
        for _ in pbar:
            loss = model.compute_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler_type != None:
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            run.log({"loss": loss.item(), "lr": lr})
    logger.info("Training finished")
    run.finish()
    
    with torch.no_grad():
        centroids = model.centroids.detach()
        centroids = F.normalize(centroids, dim=-1)
        chunk_size = 100
        sample_chunk_num = (centroids.size(0) + chunk_size - 1) // chunk_size
        sample_ids = set()
        logger.info("Sampling the samples")
        for sid in range(sample_chunk_num):
            beg = sid * chunk_size
            end = min((sid + 1) * chunk_size, centroids.size(0))
            chunk = centroids[beg: end]
            dist = torch.matmul(chunk, features.t()) # (chunk_size, n)
            _, ids = torch.sort(dist, dim=-1, descending=True)
            for i in range(ids.size(0)):
                for j in range(ids.size(1)):
                    if ids[i, j].item() not in sample_ids:
                        sample_ids.add(ids[i, j].item())
                        break
    logger.info(f"Sampled {len(sample_ids)} samples")
    sample_ids = list(sample_ids)
    sample_ids.sort()
    model.config.sample_ids = sample_ids
    model.config.save(f"{save_dir}/config.yaml")
    return sample_ids
            
            
    
    
    