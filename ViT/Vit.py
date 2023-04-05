import augly.image as imaugs
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModel, AutoFeatureExtractor
from utils.disc21 import DISC21Definition, DISC21
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="google/vit-large-patch16-224")
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--image_dir', type=str, default='/scratch/lustre/home/auma4493/images/DISC21')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--start_epoch', required=True, type=int)
    parser.add_argument('--end_epoch', required=True, type=int)
    parser.add_argument('--num_negatives', type=int, default=8)

    args = parser.parse_args()

    extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    saved_states = torch.load(args.model_ckpt)
    model.load_state_dict(saved_states['model_state_dict'])

    transformation_chain = transforms.Compose(
        [
            # We first resize the input image to 256x256, and then we take center crop.
            transforms.Resize(int((256 / 224) * 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std),
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
            imaugs.OneOf([
                imaugs.OverlayEmoji(),
                imaugs.OverlayStripes(),
                imaugs.OverlayText(),
            ], p=0.5),
            # We first resize the input image to 256x256, and then we take center crop.
            transforms.Resize(int((256 / 224) * 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std),
        ]
    )

    train_df = DISC21Definition(args.image_dir)
    train_ds = DISC21(train_df, subset='train', transform=transformation_chain, augmentations=augmentation_chain)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)

    lr = 1e-5  # could use a scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(saved_states['optimizer_state_dict'])
    loss_func = torch.nn.TripletMarginLoss(margin=0.3, p=2)

    model.train()
    print("num_negatives", args.batch_size * args.num_negatives)
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), desc="Epochs"):
        running_loss = []
        for step, (anchor_img, positive_img, index, anchor_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            pos_negatives = train_ds.get_negatives(index.numpy(), num_negatives=args.batch_size * args.num_negatives)
            negative_img = pos_negatives.to(device)
            negative_out = model(negative_img).last_hidden_state
            del pos_negatives, negative_img

            anchor_img = anchor_img.to(device)
            anchor_out = model(anchor_img).last_hidden_state
            del anchor_img

            with torch.no_grad():
                neg_matrix = torch.cdist(torch.flatten(anchor_out, start_dim=1),
                                         torch.flatten(negative_out, start_dim=1))
            negative_out = negative_out[torch.argmin(neg_matrix, dim=1)]

            positive_img = positive_img.to(device)
            positive_out = model(positive_img).last_hidden_state
            del positive_img

            loss = loss_func(anchor_out, positive_out, negative_out)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args.end_epoch, np.mean(running_loss)))
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, f"vit_checkpoints/trained_model_{epoch + 1}_{args.end_epoch}.pth")
