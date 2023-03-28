import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet101_Weights
from tqdm.auto import tqdm
from utils.disc21 import DISC21Definition, DISC21

if __name__ == '__main__':
    model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()

    transformation_chain = transforms.Compose(
        [
            # We first resize the input image to 256x256, and then we take center crop.
            transforms.Resize(int((256 / 224) * 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    df = DISC21Definition('/media/augustinas/T7/DISC2021/SmallData/images/')
    train_ds = DISC21(df, subset='train', transform=transformation_chain, augmentations=None)
    gallery_ds = DISC21(df, subset='gallery', transform=transformation_chain, augmentations=None)
    query_ds = DISC21(df, subset='query', transform=transformation_chain, augmentations=None)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)
    gallery_loader = DataLoader(gallery_ds, batch_size=1, shuffle=False, num_workers=1)
    query_loader = DataLoader(query_ds, batch_size=1, shuffle=False, num_workers=1)

    for step, (anchor, _, _, name) in enumerate(tqdm(train_loader, desc='Train', leave=False)):
        with torch.no_grad():
            anchor_out = model(anchor)
        bottleneck_values = np.squeeze(anchor_out)
        bottleneck_values = bottleneck_values / np.linalg.norm(bottleneck_values)
        np.save(f'./resnet_data/t/{name[0][:-4]}.npy', bottleneck_values)

    for step, (anchor, name) in enumerate(tqdm(gallery_loader, desc='References', leave=False)):
        with torch.no_grad():
            anchor_out = model(anchor)
        bottleneck_values = np.squeeze(anchor_out)
        bottleneck_values = bottleneck_values / np.linalg.norm(bottleneck_values)
        np.save(f'./resnet_data/r/{name[0][:-4]}.npy', bottleneck_values)

    for step, (anchor, name) in enumerate(tqdm(query_loader, desc='Queries', leave=False)):
        with torch.no_grad():
            anchor_out = model(anchor)
        bottleneck_values = np.squeeze(anchor_out)
        bottleneck_values = bottleneck_values / np.linalg.norm(bottleneck_values)
        np.save(f'./resnet_data/q/{name[0][:-4]}.npy', bottleneck_values)