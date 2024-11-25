from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from PIL import Image

# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, prompts):
        self.images = images
        self.prompts = prompts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.prompts[idx]
