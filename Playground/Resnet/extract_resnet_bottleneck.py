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
    saved_states = torch.load("/scratch/lustre/home/auma4493/TheNextModel/Runable/Resnet/resnet101_checkpoints/trained_model_10_10.pth")
    model.load_state_dict(saved_states['model_state_dict'])
    print('Loaded the model')
    
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    transformation_chain = transforms.Compose(
        [
            # We first resize the input image to 256x256, and then we take center crop.
            transforms.Resize(int((256 / 224) * 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    df = DISC21Definition('/scratch/lustre/home/auma4493/images/DISC21')
    train_ds = DISC21(df, subset='train', transform=transformation_chain, augmentations=None)
    gallery_ds = DISC21(df, subset='gallery', transform=transformation_chain, augmentations=None)
    query_ds = DISC21(df, subset='query', transform=transformation_chain, augmentations=None)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    gallery_loader = DataLoader(gallery_ds, batch_size=1, shuffle=False)
    query_loader = DataLoader(query_ds, batch_size=1, shuffle=False)

    for step, (anchor, _, _, name) in enumerate(tqdm(train_loader, desc='Train', leave=False)):
        anchor = anchor.to(device)
        with torch.no_grad():
            anchor_out = model(anchor)
        bottleneck_values = np.squeeze(anchor_out.cpu())
        bottleneck_values = bottleneck_values / np.linalg.norm(bottleneck_values)
        np.save(f'./resnet_data/10/t/{name[0][:-4]}.npy', bottleneck_values)

    for step, (anchor, name) in enumerate(tqdm(gallery_loader, desc='References', leave=False)):
        anchor = anchor.to(device)
        with torch.no_grad():
            anchor_out = model(anchor)
        bottleneck_values = np.squeeze(anchor_out.cpu())
        bottleneck_values = bottleneck_values / np.linalg.norm(bottleneck_values)
        np.save(f'./resnet_data/10/r/{name[0][:-4]}.npy', bottleneck_values)

    for step, (anchor, name) in enumerate(tqdm(query_loader, desc='Queries', leave=False)):
        anchor = anchor.to(device)
        with torch.no_grad():
            anchor_out = model(anchor)
        bottleneck_values = np.squeeze(anchor_out.cpu())
        bottleneck_values = bottleneck_values / np.linalg.norm(bottleneck_values)
        np.save(f'./resnet_data/10/q/{name[0][:-4]}.npy', bottleneck_values)