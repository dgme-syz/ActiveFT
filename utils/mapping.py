from collections import OrderedDict


MAPPING_NAME_TO_HUB = OrderedDict([
    ('CIFAR10', 'uoft-cs/cifar10'), 
])

MAPPING_NAME_TO_KEY = OrderedDict([
    ('CIFAR10', ('img', 'label')),  
])



def get_info(name: str):
    r"""
        Get the dataset information
        
        Args:
            name ('str'): 
                The name of the dataset
        Returns:
            The hub name, the image key, and the label key
    """
    return MAPPING_NAME_TO_HUB[name], *MAPPING_NAME_TO_KEY[name]