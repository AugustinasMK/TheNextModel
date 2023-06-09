import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModel, AutoImageProcessor
from utils.glv2 import GLV2Definition, GLV2
from utils.augmentation_chain import get_augmentation_chain
from utils.ggem import GGeM
import argparse
from utils.QuadrupletMarginLoss import QuadrupletMarginLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="google/vit-large-patch16-224")
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--image_dir', type=str, default='/scratch/lustre/home/auma4493/images/LANDV2')
    parser.add_argument('--labels_file', type=str, default='./labels.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--start_epoch', required=True, type=int)
    parser.add_argument('--end_epoch', required=True, type=int)
    parser.add_argument('--num_negatives', type=int, default=6)
    parser.add_argument('--use_hnm', type=str, default='False')

    args = parser.parse_args()

    print('Will load this checkpoint:', args.model_ckpt)
    print('start_epoch', args.start_epoch)
    print('end_epoch', args.end_epoch)

    use_hnm = True if args.use_hnm == 'True' else False

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.pooler = GGeM(groups=16, eps=1e-6)
    saved_states = torch.load(args.model_ckpt)
    model.load_state_dict(saved_states['model_state_dict'])

    transformation_chain = transforms.Compose(
        [
            # We first resize the input image to 256x256, and then we take center crop.
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    augmentation_chain = get_augmentation_chain(image_path=args.image_dir + '/train/', mean=processor.image_mean,
                                                std=processor.image_std)

    train_df = GLV2Definition(args.image_dir, args.labels_file)
    train_ds = GLV2(train_df, subset='train', transform=transformation_chain, augmentations=augmentation_chain,
                    use_hnm=use_hnm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)

    lr = 1e-5  # could use a scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(saved_states['optimizer_state_dict'])
    
    loss_func = QuadrupletMarginLoss(margin=1.0, alpha=0.3, p=2)

    if use_hnm:
        print("Using HNM")
        model.train()
        print("num_negatives", args.batch_size * args.num_negatives)
        for epoch in tqdm(range(args.start_epoch, args.end_epoch), desc="Epochs"):
            running_loss = []
            for step, (anchor_img, positive_img, semipositive_img, index, anchor_label) in enumerate(
                    tqdm(train_loader, desc="Training", leave=False)):
                pos_negatives = train_ds.get_negatives(index.numpy(),
                                                       num_negatives=args.batch_size * args.num_negatives)
                negative_img = pos_negatives.to(device)
                negative_out = model(negative_img).pooler_output
                del pos_negatives, negative_img

                anchor_img = anchor_img.to(device)
                anchor_out = model(anchor_img).pooler_output
                del anchor_img

                with torch.no_grad():
                    neg_matrix = torch.cdist(torch.flatten(anchor_out, start_dim=1),
                                             torch.flatten(negative_out, start_dim=1))
                negative_out = negative_out[torch.argmin(neg_matrix, dim=1)]

                positive_img = positive_img.to(device)
                positive_out = model(positive_img).pooler_output
                del positive_img

                semipositive_img = semipositive_img.to(device)
                semipositive_out = model(semipositive_img).pooler_output
                del semipositive_img

                loss = loss_func(anchor_out, positive_out, semipositive_out, negative_out)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss.append(loss.cpu().detach().numpy())
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args.end_epoch, np.mean(running_loss)))
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                        }, f"checkpoints4/trained_model_{epoch + 1}_{args.end_epoch}.pth")
    else:
        print("Not using HNM")
        model.train()
        print("batch_size", args.batch_size)
        print("sanity check GLV2 4")
        for epoch in tqdm(range(args.start_epoch, args.end_epoch), desc="Epochs"):
            running_loss = []
            for step, (
                    anchor_img, positive_img, semipositive_img, negative_img, anchor_name, anchor_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
                anchor_img = anchor_img.to(device)
                positive_img = positive_img.to(device)
                semipositive_img = semipositive_img.to(device)
                negative_img = negative_img.to(device)

                anchor_out = model(anchor_img).pooler_output
                del anchor_img
                positive_out = model(positive_img).pooler_output
                del positive_img
                semipositive_out = model(semipositive_img).pooler_output
                del semipositive_img
                negative_out = model(negative_img).pooler_output
                del negative_img

                loss = loss_func(anchor_out, positive_out, semipositive_out, negative_out)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss.append(loss.cpu().detach().numpy())
                if step % 1_000 == 0:
                    print("Iteration: {} - Loss: {:.4f}".format(step + 1, np.mean(running_loss)))
                    print(loss)
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args.end_epoch, np.mean(running_loss)))
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                        }, f"checkpoints4/trained_model_{epoch + 1}_{args.end_epoch}.pth")
