import os
import csv
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np

def get_files_in_dir(dir_path, extensions = [], recursive = False, absolute_path = False):
    """Returns list of all files in directory.

    Args:
        dir_path: Path to directory.
        extensions: optionally return only files with certain extensions (case insensitive).
        recursive: Also returns files in subdirectories.
        absolute_path: Returns absolute path of the files (if False, returns relative path from dir_path)
    """
    if not os.path.exists(dir_path):
        raise OSError(f'Directory does not exist: {dir_path}')

    lower_extensions = [ext.lower() for ext in extensions]

    file_list = []

    for root, dirs, files in tqdm(os.walk(dir_path)):
        if not recursive:
            # stop recursion
            dirs.clear()

        for f in tqdm(files):
            if lower_extensions:
                _, ext = os.path.splitext(f)
                if ext.lower() not in lower_extensions:
                    continue

            full_path = os.path.join(root, f)
            if absolute_path:
                file_list.append(os.path.abspath(full_path))
            else:
                file_list.append(os.path.relpath(full_path, dir_path))

    return file_list


def main():
    progress_file = '/scratch/lustre/home/auma4493/TheNextModel/Runable/progress.txt'
    ptime = {}

    query_dir = '/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/10/q'
    p_start_time = datetime.now()
    # Load query list
    print("Query directory: {}".format(query_dir))
    query_list = get_files_in_dir(query_dir, extensions=['.npy'])
    query_list = sorted(query_list)
    print("Found {} query files".format(len(query_list)))
    ptime['query_list'] = (datetime.now() - p_start_time).total_seconds()

    reference_dir = '/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/10/r'
    p_start_time = datetime.now()
    # Load reference list
    print("Reference directory: {}".format(reference_dir))
    reference_list = get_files_in_dir(reference_dir, extensions=['.npy'])
    reference_list = sorted(reference_list)
    print("Found {} reference files".format(len(reference_list)))
    ptime['reference_list'] = (datetime.now() - p_start_time).total_seconds()

    train_dir = '/scratch/lustre/home/auma4493/TheNextModel/Runable/resnet_data/10/t'
    p_start_time = datetime.now()
    # Load train list
    print("Train directory: {}".format(train_dir))
    train_list = get_files_in_dir(train_dir, extensions=['.npy'])
    train_list = sorted(train_list)
    print("Found {} train files".format(len(train_list)))
    ptime['train_list'] = (datetime.now() - p_start_time).total_seconds()

    p_start_time = datetime.now()
    # Load query features
    query_count = len(query_list)
    query_bottlenecks = np.empty((query_count, 100352), dtype=np.float32)
    for i, fn in enumerate(tqdm(query_list, desc='Queries')):
        feat = np.load(os.path.join(query_dir, fn))
        query_bottlenecks[i] = feat
    print("Loaded {} query features".format(query_count))
    ptime['query_bottlenecks'] = (datetime.now() - p_start_time).total_seconds()

    p_start_time = datetime.now()
    # Load reference features
    reference_count = len(reference_list)
    reference_bottlenecks = np.empty((reference_count, 100352), dtype=np.float32)
    for i, fn in enumerate(tqdm(reference_list, desc='References')):
        feat = np.load(os.path.join(reference_dir, fn))
        reference_bottlenecks[i] = feat
    print("Loaded {} reference features".format(reference_count))
    ptime['reference_bottlenecks'] = (datetime.now() - p_start_time).total_seconds()

#     p_start_time = datetime.now()
#     # Load training features
#     train_count = len(train_list)
#     train_bottlenecks = np.empty((train_count, 100352), dtype=np.float32)
#     for i, fn in enumerate(tqdm(train_list, desc='Train')):
#         feat = np.load(os.path.join(train_dir, fn))
#         train_bottlenecks[i] = feat
#     print("Loaded {} train features".format(train_count))
#     ptime['train_bottlenecks'] = (datetime.now() - p_start_time).total_seconds()

#     norms = np.dot(query_bottlenecks, train_bottlenecks.T)
#     norms = np.mean(norms, axis=1)

    p_start_time = datetime.now()
    # Calculate similarity
    similarity = np.dot(query_bottlenecks, reference_bottlenecks.T)
    # similarity = similarity - norms[:, None]

    print('Sorting indices')
    sorted_indices = np.argsort(similarity.flatten())[-500_000:][::-1]
    print(sorted_indices[:10])
    print('Sorted indices')

    print('Unravel index')
    # highest_elements_with_indexes = np.unravel_index(sorted_indices, similarity.shape)
    # print('balys', highest_elements_with_indexes[:10])

    print('Writing to file')
    with open('resnet_no_norm_500k.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['query_id', 'reference_id', 'score'])
        for idx in tqdm(sorted_indices):
            row, col = np.unravel_index(idx, similarity.shape)
            query = f'Q{row:04d}'
            reference = f'R{col:05d}'
            score = similarity[row, col]
            writer.writerow([query, reference, score])
    print('Done')

    

if __name__ == '__main__':
    main()
