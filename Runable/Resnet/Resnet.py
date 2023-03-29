import augly.image as imaugs
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm
from torchvision.models import ResNet101_Weights
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

    augmentation_chain = transforms.Compose(
        [
            imaugs.Brightness(factor=2.0),
            imaugs.RandomRotation(),
            imaugs.OneOf([
                imaugs.RandomAspectRatio(),
                imaugs.RandomBlur(),
                imaugs.RandomBrightness(),
                imaugs.RandomNoise(),
                imaugs.RandomPixelization(),
            ]),
            # We first resize the input image to 256x256, and then we take center crop.
            transforms.Resize(int((256 / 224) * 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_df = DISC21Definition('/scratch/lustre/home/auma4493/images/DISC21')
    train_ds = DISC21(train_df, subset='train', transform=transformation_chain, augmentations=augmentation_chain)

    embedding_dims = 2
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)

    epoch_count = 10  # for now
    lr = 1e-5  # could use a scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.TripletMarginLoss()

    model.train()
    for epoch in tqdm(range(epoch_count), desc="Epochs"):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            loss = loss_func(anchor_out, positive_out, negative_out)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epoch_count, np.mean(running_loss)))
        torch.save({"model_state_dict": model.state_dict(),
                    "optimzier_state_dict": optimizer.state_dict()
                    }, f"resnet101_checkpoints/trained_model_{epoch + 1}_{epoch_count}.pth")
