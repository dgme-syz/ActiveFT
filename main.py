from datasets import load_dataset
from transformers import ViTModel
from torch.utils.data import DataLoader
import logging
import torch
from tqdm import tqdm
import numpy as np
import os
from typing import Literal
from utils import get_info
import torchvision.transforms as T
from concurrent.futures import ThreadPoolExecutor
from functools import partial


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

tfm = T.Compose([
    T.Resize((224, 224), interpolation=3),
    T.ToTensor(), 
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def transform_fn(
    examples: dict,
    dataset_name: str = "CIFAR10"
) -> dict:
    r"""
        use multithread to transform images
        
        Args:
            examples ('dict'):
                the '__getitem__' output of datasets object
            dataset_name ('str', defaults to CIFAR10):
                used to extract two important keys 
        Returns:
            when we use data iter and it call '__getitem__' of dataset, we will return the transformed dict
    """
    _, img_key, label_key = get_info(dataset_name)
    
    pixel_values, label = [], []
    with ThreadPoolExecutor() as executor:
        for x in examples[img_key]:
            pixel_values.append(executor.submit(tfm, x))
        label.extend(examples[label_key])
    assert len(pixel_values) == len(label)
    return {
        'img': [x.result() for x in pixel_values],
        'label': label
    }

def extract_features(
    batch_size: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    n_last_blocks: int = 4,
    avgpool_patchtokens: bool = False,
    save_dir: str = None,
    dataset_name: Literal['CIFAR10', 'CIFAR100'] = 'CIFAR10',
    num_workers: int = 2
):
    r"""
        We only consider using dino-vits16 and cifar-10 for the feature extraction task
        
        Args:
            batch_size ('int', defaults to 128): 
                The batch size for the data loader
            device ('str', defaults to 'cuda' if available else 'cpu'): 
                The device to run the model on
            n_last_blocks ('int', defaults to 4): 
                Concatenate [CLS] tokens for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.
            avgpool_patchtokens ('bool', defaults to False):
                If True, average pool the output of the patch tokens in the last block
            save_dir ('str', defaults to None):
                The directory to save the extracted features
            dataset_name ('str', defaults to 'CIFAR10'):
                The dataset name to use for the feature extraction task
            num_workers ('int', defaults to 4):
                The number of workers to use for the data loader
    """
    model = ViTModel.from_pretrained(
        'facebook/dino-vits16', add_pooling_layer=False).eval().to(device)
    hub, _, _ = get_info(dataset_name)
    dataset = load_dataset(hub, split='train')
        
    dataset.set_transform(partial(transform_fn, dataset_name=dataset_name))
    data_iter = DataLoader(
        dataset, 
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )    
    
    logger.info(f"{model._get_name()} loaded, extracting features for {len(dataset)} images")
    
    features, targets = (), ()
    with torch.no_grad():
        for batch in tqdm(data_iter):
            inp = batch['img'].to(device)
            assert inp.size(1) == 3 and len(inp.size()) == 4, "Input shape must be (B, C, H, W)"
            intermediate_output = model(inp, output_hidden_states=True)["hidden_states"][-n_last_blocks:]
            
            out = torch.cat(
                [x[:, 0] for x in intermediate_output], dim=-1
            ) # Concatenate [CLS] tokens for the `n` last blocks
            if avgpool_patchtokens:
                out = torch.cat(
                    (out.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1
                )
                out = out.reshape(out.size(0), -1)
                
            out = out.cpu().numpy()
            tgt = batch['label'].numpy()
            
            targets += (tgt,)
            features += (out,)
    logger.info(f"Finished extracting")
    save_outputs = np.concatenate(
        (np.concatenate(features, axis=0), np.concatenate(targets, axis=0).reshape(-1, 1)), axis=1
    )
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(
        save_dir, f"cifar10_train.npy"
    ), save_outputs)
    logger.info(f"Features extracted, saving to {save_dir}")
    
    
if __name__ == "__main__":
    extract_features(
        batch_size=128,
        device='cuda',
        n_last_blocks=4,
        avgpool_patchtokens=False,
        save_dir='features',
        dataset_name='CIFAR10',
        num_workers=2 
    )