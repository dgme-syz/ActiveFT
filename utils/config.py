from dataclasses import dataclass, field
import torch
import yaml

@dataclass
class FTConfig(object):
    sample_num: int = field(
        default=100, 
        metadata={"help": "The number of the samples"}, 
    )
    
    temperature: float = field(
        default=0.07, 
        metadata={"help": "The temperature of the softmax"}, 
    )
    
    balance: float = field(
        default=1.0, 
        metadata={"help": "The balance of the loss"}, 
    )
    
    slice: int = field(
        default=None, 
        metadata={"help": "The slice used for similarity calculation"}, 
    )
    
    batch_size: int = field(
        default=100000, 
        metadata={"help": "The batch size for training"}, 
    )
    
    device: str = field(
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        metadata={"help": "The device to run the model on"}, 
    )
    
    sample_ids: list = field(
        default=None, 
        metadata={"help": "The sample ids"}, 
    )
    
    def save(self, path: str):
        r"""
            Save the configuration to a file
        """
        with open(path, 'w') as f:
            yaml.safe_dump(self.__dict__, f)
            
    @classmethod
    def load(cls, path: str):
        r"""
            Load the configuration from a file
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def __repr__(self):
        return str(self.__dict__)
    
    