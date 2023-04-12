from datasets import load_dataset
from transformers import  AutoImageProcessor, AutoModel
from tqdm.auto import tqdm
import torchvision.transforms
import torch
import numpy as np

def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        image_batch_transformed = torch.stack(
            [transformation_chain(image) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

    return pp


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.cpu().numpy().tolist()


batch_size = 32

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("imagefolder", name="disc21-next-vit", data_dir="/scratch/lustre/home/auma4493/images/DISC21/",
                       drop_labels=True)
    

    # Load the model
    model_ckpt = "google/vit-large-patch16-224"
    processor =  AutoImageProcessor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    saved_states = torch.load("/scratch/lustre/home/auma4493/TheNextModel/ViT/vit_checkpoints/trained_model_16_20.pth")
    model.load_state_dict(saved_states['model_state_dict'])

    # Create the transform
    transformation_chain = torchvision.transforms.Compose(
        [
            # We first resize the input image to 256x256, and then we take center crop.
            torchvision.transforms.Resize(int((256 / 224) * processor.size["height"])),
            torchvision.transforms.CenterCrop(processor.size["height"]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    # Create the preprocessing function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device))

    # Compute the embeddings
    query_emb = dataset["test"].map(extract_fn, batched=True, batch_size=batch_size)
    references_emb = dataset["validation"].map(extract_fn, batched=True, batch_size=batch_size)
    train_emb = dataset["train"].map(extract_fn, batched=True, batch_size=batch_size)

    # Make numpy arrays
    query_embeddings = np.array(query_emb["embeddings"])
    query_embeddings = torch.from_numpy(query_embeddings).cuda()
    print('query_embeddings', query_embeddings)
    reference_embeddings = np.array(references_emb["embeddings"])
    reference_embeddings = torch.from_numpy(reference_embeddings).cuda()
    train_embeddings = np.array(train_emb["embeddings"])
    train_embeddings = torch.from_numpy(train_embeddings).cuda()
    
    print(train_embeddings[0].shape)

    # Compute norms
    norms = []
    for i in tqdm(range(len(query_embeddings))):
        norms.append(compute_scores(train_embeddings, query_embeddings[i]))
    norms = np.array(norms)
    norms = np.mean(norms, axis=1)

    matrix = []
    for i in tqdm(range(len(query_embeddings))):
        sim_scores = compute_scores(reference_embeddings, query_embeddings[i])
        matrix.append(sim_scores)
    matrix = np.array(matrix)
    print(matrix)
    print(matrix.shape)
    np.save("disc_matrix_no_norm.npy", matrix)

    # Normalize the matrix
    matrix = matrix - norms[:, None]
    print(matrix)
    print(matrix.shape)
    # Save the matrix
    np.save("disc_matrix_norm.npy", matrix)