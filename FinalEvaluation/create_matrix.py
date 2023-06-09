import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModel, AutoImageProcessor

from utils.compute_scores import compute_scores
from utils.extract_embeddings import extract_embeddings
from utils.poolers import GGeM, ClassToken
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_name', type=str, default="google/vit-large-patch16-224")
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--use_GeM', action='store_true')
    parser.add_argument('-m', '--model_type', type=str, default='disc', choices=['disc', 'glv2_q', 'glv2_t', 'glv2_s'])
    parser.add_argument('--model_checkpoint', type=str, required=True)

    # Dataset
    parser.add_argument('-d', '--dataset', type=str, default='disc', choices=['disc', 'glv2_q', 'glv2_t', 'oxpa'])

    args = parser.parse_args()
    print("args:", args)

    # Load dataset
    if args.dataset == 'disc':
        dataset = load_dataset("imagefolder", name="disc21-next-final",
                               data_dir="/scratch/lustre/home/auma4493/images/DISC21/", drop_labels=True)
    elif args.dataset == 'oxpa':
        dataset = load_dataset("imagefolder", name="oxpa-next-final",
                               data_dir="/scratch/lustre/home/auma4493/images/OxPa/", drop_labels=True)
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

    if args.dataset == 'disc':
        save_dir = f"./data/disc/{args.model_type}/{saved_states['epoch']}/"
    elif args.dataset == 'glv2_q':
        save_dir = f"./data/glv2_q/{args.model_type}/{saved_states['epoch']}/"
    elif args.dataset == 'oxpa':
        save_dir = f"./data/oxpa/{args.model_type}/{saved_states['epoch']}/"
    else:
        save_dir = f"./data/glv2_t/{args.model_type}/{saved_states['epoch']}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the preprocessing function
    extract_fn = extract_embeddings(model.to(device), transformation_chain)

    # Compute the embeddings
    query_emb = dataset["test"].map(extract_fn, batched=True, batch_size=args.batch_size)
    # train_emb = train_dataset["train"].map(extract_fn, batched=True, batch_size=args.batch_size)

    # Make numpy arrays
    query_embeddings = np.array(query_emb["embeddings"])
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings = torch.from_numpy(query_embeddings).cuda()
    print('query_embeddings', query_embeddings)
#     train_embeddings = np.array(train_emb["embeddings"])
#     train_embeddings = torch.from_numpy(train_embeddings).cuda()
#     print(train_embeddings[0].shape)

#     # Compute norms
#     norms = []
#     for i in tqdm(range(len(query_embeddings)), desc='Computing norms'):
#         norms.append(compute_scores(train_embeddings, query_embeddings[i]))
#     norms = np.array(norms)
#     norms = np.mean(norms, axis=1)
#     np.save(f"{save_dir}norms.npy", norms)

    # Make numpy arrays
    # del train_embeddings, train_emb
    references_emb = dataset["validation"].map(extract_fn, batched=True, batch_size=args.batch_size)
    reference_embeddings = np.array(references_emb["embeddings"])
    reference_embeddings /= np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
    reference_embeddings = torch.from_numpy(reference_embeddings).cuda()
    print(reference_embeddings[0].shape)

    # Compute the matrix
    matrix = []
    start_time = datetime.now()
    for i in range(len(query_embeddings)):
        sim_scores = compute_scores(reference_embeddings, query_embeddings[i])
        matrix.append(sim_scores)
    end_time = datetime.now()
    secs = (end_time - start_time).total_seconds()
    print("BUYBBIYB:", secs)
    matrix = np.array(matrix)
    print(matrix)
    print(matrix.shape)
    np.save(f"{save_dir}matrix_no_norm.npy", matrix)

    # # Normalize the matrix
    # matrix = matrix - norms[:, None]
    # print(matrix)
    # print(matrix.shape)
    # # Save the matrix
    # np.save(f"{save_dir}matrix_norm.npy", matrix)
