import torch
from torchvision import transforms

class EventDataAugmentation:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transform = transforms.Compose([
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomApply([transforms.Lambda(self.add_gaussian_noise)], p=0.5)
        ])

    def add_gaussian_noise(self, tensor, mean=0, std=0.01):
        noise = torch.randn(tensor.size()) * std + mean
        return tensor + noise

    def __call__(self, event_volume):
        return self.transform(event_volume)
