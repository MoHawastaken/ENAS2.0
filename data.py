import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Image(object):
    def __init__(self, batch_size, test_batch_size, mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768], preprocesses=[], num_workers=1):
        Dataset = datasets.CIFAR10
        
        normalize = transforms.Normalize(mean, std)
        
        # preprocessing of training data
        transform = transforms.Compose(preprocesses + [
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.train = t.utils.data.DataLoader(
            Dataset(root='./data', train=True, transform=transform, download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        self.test = t.utils.data.DataLoader(
            Dataset(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
