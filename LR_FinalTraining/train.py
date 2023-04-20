import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModel, AutoImageProcessor

from utils.QuadrupletMarginLoss import QuadrupletMarginLoss
from utils.augmentation_chain import get_augmentation_chain
from utils.disc21 import DISC21Definition, DISC21
from utils.glv2 import GLV2Definition, GLV2
from utils.poolers import GGeM, ClassToken
from utils.scheduler import cosine_lr


def triplet_hnm(anchor_img, batch_size, index, num_negatives, positive_img):
    global train_ds, device, model, loss_func, optimizer, running_loss
    pos_negatives = train_ds.get_negatives(index.numpy(), num_negatives=batch_size * num_negatives)
    negative_img = pos_negatives.to(device)
    negative_out = model(negative_img).pooler_output  # (batch_size * num_negatives, 1024)
    del pos_negatives, negative_img
    anchor_img = anchor_img.to(device)
    anchor_out = model(anchor_img).pooler_output  # (batch_size, 1024)
    del anchor_img
    with torch.no_grad():
        neg_matrix = torch.cdist(torch.flatten(anchor_out, start_dim=1),
                                 torch.flatten(negative_out, start_dim=1))
    negative_out = negative_out[torch.argmin(neg_matrix, dim=1)]  # (batch_size, 1024)
    positive_img = positive_img.to(device)
    positive_out = model(positive_img).pooler_output  # (batch_size, 1024)
    del positive_img
    loss = loss_func(anchor_out, positive_out, negative_out)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    running_loss.append(loss.cpu().detach().numpy())


def quad_hnm(anchor_img, batch_size, index, num_negatives, positive_img, semipositive_img):
    global train_ds, device, model, loss_func, optimizer, running_loss
    pos_negatives = train_ds.get_negatives(index.numpy(), num_negatives=batch_size * num_negatives)
    negative_img = pos_negatives.to(device)
    negative_out = model(negative_img).pooler_output  # (batch_size * num_negatives, 1024)
    del pos_negatives, negative_img
    anchor_img = anchor_img.to(device)
    anchor_out = model(anchor_img).pooler_output  # (batch_size, 1024)
    del anchor_img
    with torch.no_grad():
        neg_matrix = torch.cdist(torch.flatten(anchor_out, start_dim=1),
                                 torch.flatten(negative_out, start_dim=1))
    negative_out = negative_out[torch.argmin(neg_matrix, dim=1)]  # (batch_size, 1024)
    positive_img = positive_img.to(device)
    positive_out = model(positive_img).pooler_output  # (batch_size, 1024)
    del positive_img
    semipositive_img = semipositive_img.to(device)
    semipositive_out = model(semipositive_img).pooler_output
    del semipositive_img
    loss = loss_func(anchor_out, positive_out, semipositive_out, negative_out)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    running_loss.append(loss.cpu().detach().numpy())


def run_epoch_with_hnm(batch_size, num_negatives, dataset, print_freq=1000):
    global running_loss
    print("Using HNM")
    print("num_negatives", batch_size * num_negatives)
    if dataset == 'disc':
        for step, (anchor_img, positive_img, index, anchor_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            triplet_hnm(anchor_img, batch_size, index, num_negatives, positive_img)
            if step % print_freq == 0:
                print("Iteration: {} - Loss: {:.4f}".format(step + 1, np.mean(running_loss)))
                print(running_loss[-1])
    else:
        for step, (anchor_img, positive_img, semipositive_img, index, anchor_name, anchor_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            if dataset == 'glv2_q':
                quad_hnm(anchor_img, batch_size, index, num_negatives, positive_img, semipositive_img)
            else:
                triplet_hnm(anchor_img, batch_size, index, num_negatives, positive_img)
            if step % print_freq == 0:
                print("Iteration: {} - Loss: {:.4f}".format(step + 1, np.mean(running_loss)))
                print(running_loss[-1])


def triplet(anchor_img, positive_img, negative_img):
    global device, model, loss_func, optimizer, running_loss
    anchor_img = anchor_img.to(device)
    positive_img = positive_img.to(device)
    negative_img = negative_img.to(device)

    anchor_out = model(anchor_img).pooler_output  # (batch_size, 1024)
    positive_out = model(positive_img).pooler_output  # (batch_size, 1024)
    negative_out = model(negative_img).pooler_output  # (batch_size, 1024)

    loss = loss_func(anchor_out, positive_out, negative_out)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    running_loss.append(loss.cpu().detach().numpy())


def quad(anchor_img, positive_img, semipositive_img, negative_img):
    global device, model, loss_func, optimizer, running_loss
    anchor_img = anchor_img.to(device)
    positive_img = positive_img.to(device)
    semipositive_img = semipositive_img.to(device)
    negative_img = negative_img.to(device)

    anchor_out = model(anchor_img).pooler_output  # (batch_size, 1024)
    del anchor_img
    positive_out = model(positive_img).pooler_output  # (batch_size, 1024)
    del positive_img
    semipositive_out = model(semipositive_img).pooler_output  # (batch_size, 1024)
    del semipositive_img
    negative_out = model(negative_img).pooler_output  # (batch_size, 1024)
    del negative_img

    loss = loss_func(anchor_out, positive_out, semipositive_out, negative_out)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    running_loss.append(loss.cpu().detach().numpy())


def run_epoch_without_hnm(dataset, print_freq=1000):
    global running_loss, args
    print("Not using HNM")
    print("batch_size", args.batch_size)
    if dataset == 'disc':
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            triplet(anchor_img, positive_img, negative_img)
            if step % print_freq == 0:
                print("Iteration: {} - Loss: {:.4f}".format(step + 1, np.mean(running_loss)))
                print(running_loss[-1])
    else:
        for step, (anchor_img, positive_img, semipositive_img, negative_img, anchor_name, anchor_label) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            if dataset == 'glv2_q':
                quad(anchor_img, positive_img, semipositive_img, negative_img)
            else:
                triplet(anchor_img, positive_img, negative_img)
            if step % print_freq == 0:
                print("Iteration: {} - Loss: {:.4f}".format(step + 1, np.mean(running_loss)))
                print(running_loss[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_name', type=str, default="google/vit-large-patch16-224")
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--num_negatives', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)

    # Dataset
    parser.add_argument('-d', '--dataset', type=str, default='disc', choices=['disc', 'glv2_q', 'glv2_t'])

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters, for pretrained ")

    # Training
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--labels_file', type=str, default='./glv2_labels.csv')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--use_hnm', action='store_true')
    parser.add_argument('--use_GeM', action='store_true')
    parser.add_argument('--print-freq', type=int, default=1000)

    args = parser.parse_args()
    print(args)

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    if args.use_GeM:
        model.pooler = GGeM(groups=16, eps=1e-6)
    else:
        model.pooler = ClassToken()

    print(model)

    transformation_chain = transforms.Compose(
        [
            # We first resize the input image to 256x256, and then we take center crop.
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    if args.dataset == 'disc':
        image_dir = '/scratch/lustre/home/auma4493/images/DISC21'
    else:
        image_dir = '/scratch/lustre/home/auma4493/images/LANDV2'
    augmentation_chain = get_augmentation_chain(image_path=image_dir + '/train/', mean=processor.image_mean,
                                                std=processor.image_std)

    if args.dataset == 'disc':
        train_df = DISC21Definition(image_dir)
        train_ds = DISC21(train_df, subset='train', transform=transformation_chain, augmentations=augmentation_chain,
                          use_hnm=args.use_hnm)
    else:
        train_df = GLV2Definition(image_dir, args.labels_file)
        train_ds = GLV2(train_df, subset='train', transform=transformation_chain, augmentations=augmentation_chain,
                        use_hnm=args.use_hnm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lambda_lr = lambda ech: cosine_lr(ech)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr, verbose=True)

    start_epoch = 0
    if args.resume != '':
        print("Loading saved model")
        saved_states = torch.load(args.resume)
        model.load_state_dict(saved_states['model_state_dict'])
        optimizer.load_state_dict(saved_states['optimizer_state_dict'])
        scheduler.load_state_dict(saved_states['scheduler_state_dict'])
        start_epoch = saved_states['epoch']

    if args.dataset == 'disc':
        loss_func = torch.nn.TripletMarginLoss(margin=0.3, p=2)
        save_dir = './checkpoints/disc'
    elif args.dataset == 'glv2_t':
        loss_func = torch.nn.TripletMarginLoss(margin=0.3, p=2)
        save_dir = './checkpoints/glv2_triplet'
    else:
        loss_func = QuadrupletMarginLoss(margin=1.0, alpha=0.3, p=2)
        save_dir = './checkpoints/glv2_quad'

    model.train()
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Epochs"):
        running_loss = []
        if args.use_hnm:
            run_epoch_with_hnm(batch_size=args.batch_size, num_negatives=args.num_negatives, dataset=args.dataset,
                               print_freq=args.print_freq)
        else:
            run_epoch_without_hnm(dataset=args.dataset, print_freq=args.print_freq)
        scheduler.step()
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args.epochs, np.mean(running_loss)))
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1
                    }, f"{save_dir}/trained_model_{epoch + 1}_{args.epochs}.pth")
