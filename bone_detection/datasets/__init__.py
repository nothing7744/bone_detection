from .myDataset import build_bone
import torch
import torchvision
import torch.utils.data
import datasets
def get_bone_api_from_dataset(dataset):
    print("====================")
    print(type(dataset))
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, datasets.myDataset.BoneDetection):
        print("dataset.data")
        return dataset.dataset


def build_data(image_set,args):
    return build_bone(image_set,args) 
