import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
from torchvision import transforms, models
from torchvision.models import ResNet101_Weights

# def extract_embeddings(model: torch.nn.Module):
#     """Utility to compute embeddings."""
#     device = "cuda"

#     def pp(batch):
#         images = batch["image"]
#         image_batch_transformed = torch.stack(
#             [transformation_chain(image) for image in images]
#         )
#         new_batch = image_batch_transformed.to(device)
#         with torch.no_grad():
#             embeddings = model(new_batch).cpu()
#         return {"embeddings": embeddings}

#     return pp


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.cpu().numpy().tolist()


batch_size = 8

if __name__ == '__main__':
#     # Load the dataset
#     dataset = load_dataset("imagefolder", name="disc21-next-resnet", data_dir="/scratch/lustre/home/auma4493/images/DISC21/",
#                        drop_labels=True)
    

#     # Load the model
#     print('Load the model')
#     model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
#     model.fc = torch.nn.Identity()
#     model.avgpool = torch.nn.Identity()
#     #saved_states = torch.load("/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet101_checkpoints/trained_model_10_10.pth")
#     #model.load_state_dict(saved_states['model_state_dict'])
#     print('Loaded the model')
    
#     # Create the transform
#     transformation_chain = T.Compose(
#         [
#             # We first resize the input image to 256x256, and then we take center crop.
#             T.Resize(256),
#             T.CenterCrop(224),
#             T.ToTensor(),
#             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ]
#     )

#     # Create the preprocessing function
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     extract_fn = extract_embeddings(model.to(device))
  
#     # Compute the embeddings
#     print('Compute query_emb embeddings')
#     query_emb = dataset["test"].map(extract_fn, batched=True, batch_size=batch_size)
#     query_emb.save_to_disk("/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/q")
#     print('Compute references_emb embeddings')
#     references_emb = dataset["validation"].map(extract_fn, batched=True, batch_size=batch_size)
#     references_emb.save_to_disk("/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/r")
#     print('Compute train_emb embeddings')
#     train_emb = dataset["train"].map(extract_fn, batched=True, batch_size=batch_size)
#     train_emb.save_to_disk("/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/t")
#     print('Computed train_emb embeddings')
    
    # Make numpy arrays
    print('Make Q numpy arrays')
    query_embeddings = np.array(load_from_disk("/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/q")["embeddings"])
    query_embeddings = torch.from_numpy(query_embeddings).cuda()
    print('Make R numpy arrays')
    reference_embeddings = np.array(load_from_disk("/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/r")["embeddings"])
    reference_embeddings = torch.from_numpy(reference_embeddings).cuda()
    print('Make T numpy arrays')
    train_embeddings = np.array(load_from_disk("/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/t")["embeddings"])
    train_embeddings = torch.from_numpy(train_embeddings).cuda()
    print('Made Q numpy arrays')
    print(train_embeddings[0].shape)

    # Compute norms
    print('Compute norms')
    norms = []
    for i in tqdm(range(len(query_embeddings))):
        norms.append(compute_scores(train_embeddings, query_embeddings[i]))
    norms = np.array(norms)
    norms = np.mean(norms, axis=1)
    
    print('Compute matrix')
    matrix = []
    for i in tqdm(range(len(query_embeddings))):
        sim_scores = compute_scores(reference_embeddings, query_embeddings[i])
        matrix.append(sim_scores)
    matrix = np.array(matrix)
    print(matrix)
    print(matrix.shape)
    np.save("res_disc_matrix_no_norm.npy", matrix)
    print('Computed matrix')
    
    # Normalize the matrix
    print('Normalize')
    matrix = matrix - norms[:, None]
    print(matrix)
    print(matrix.shape)
    # Save the matrix
    np.save("res_disc_matrix_norm.npy", matrix)
    print('Normalized')