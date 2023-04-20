import argparse
import csv

import numpy as np


def create(file_path, num_of_results):
    global args
    similarity = np.load(file_path)

    print('Sorting indices')
    sorted_indices = np.argsort(similarity.flatten())[-num_of_results:][::-1]
    print(sorted_indices[:10])
    print('Sorted indices')

    print('Writing to csv')
    with open(f'{file_path[:-4]}.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['query_id', 'reference_id', 'score'])
        for idx in sorted_indices:
            row, col = np.unravel_index(idx, similarity.shape)
            query = f'Q{row:04d}'
            reference = f'R{col:05d}' if args.dataset == 'disc' else f'R{col:04d}'
            score = similarity[row, col]
            writer.writerow([query, reference, score])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default='disc', choices=['disc', 'glv2_q', 'glv2_t'])
    parser.add_argument('-e', '--epoch', type=str, required=True)

    args = parser.parse_args()

    if args.dataset == 'disc':
        save_dir = f"./data/disc/{args.epoch}/"
    elif args.dataset == 'glv2_q':
        save_dir = f"./data/glv2_q/{args.epoch}/"
    else:
        save_dir = f"./data/glv2_t/{args.epoch}/"

    create(f"{save_dir}matrix_no_norm.npy", 500_000)
    create(f"{save_dir}matrix_norm.npy", 500_000)
