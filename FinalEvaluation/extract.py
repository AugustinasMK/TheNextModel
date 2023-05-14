import argparse
import os

import torch
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor

from utils.poolers import GGeM, ClassToken

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_name', type=str, default="google/vit-large-patch16-224")
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--use_GeM', action='store_true')
    parser.add_argument('-m', '--model_type', type=str, default='disc', choices=['disc', 'glv2_q', 'glv2_t'])
    parser.add_argument('--model_checkpoint', type=str, required=True)

    # Dataset
    parser.add_argument('-d', '--dataset', type=str, default='disc', choices=['disc', 'glv2_q', 'glv2_t'])

    args = parser.parse_args()
    print("args:", args)

    # Load dataset
    if args.dataset == 'disc':
        dataset = load_dataset("imagefolder", name="disc21-next-final",
                               data_dir="/scratch/lustre/home/auma4493/images/DISC21/", drop_labels=True)
    else:
        dataset = load_dataset("imagefolder", name="glv2-next-final",
                               data_dir="/scratch/lustre/home/auma4493/images/LANDV2/", drop_labels=True)
    print("main dataset: ", dataset)

    # Load dataset
    if args.model_type == 'disc':
        train_dataset = load_dataset("imagefolder", name="disc21-next-final",
                                     data_dir="/scratch/lustre/home/auma4493/images/DISC21/", drop_labels=True)
    else:
        train_dataset = load_dataset("imagefolder", name="glv2-next-final",
                                     data_dir="/scratch/lustre/home/auma4493/images/LANDV2/", drop_labels=True)
    print("train dataset: ", train_dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    if args.use_GeM:
        model.pooler = GGeM(groups=16, eps=1e-6)
    else:
        model.pooler = ClassToken()
    saved_states = torch.load(args.model_checkpoint, map_location=torch.device(device))
    model.load_state_dict(saved_states['model_state_dict'])
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

    time = 0
    for i in range(0, 100):
        image = transformation_chain(dataset[i]['image'])
        print(dataset[i]['image'])
        print("image shape: ", image.shape)
        emb = model(image.unsqueeze(0).to(device))
