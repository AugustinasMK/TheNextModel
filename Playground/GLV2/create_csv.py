import numpy as np
import csv

similarity = np.load('/scratch/lustre/home/auma4493/TheNextModel/GLV2/glv2_matrix_norm.npy')

print('Sorting indices')
sorted_indices = np.argsort(similarity.flatten())[-500_000:][::-1]
print(sorted_indices[:10])
print('Sorted indices')

print('Unravel index')
# highest_elements_with_indexes = np.unravel_index(sorted_indices, similarity.shape)
# print('balys', highest_elements_with_indexes[:10])

print('Writing to file')
with open('/scratch/lustre/home/auma4493/TheNextModel/GLV2/glv2_norm.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['query_id', 'reference_id', 'score'])
    for idx in sorted_indices:
        row, col = np.unravel_index(idx, similarity.shape)
        print(row, col)
        query = f'Q{row:04d}'
        reference = f'R{col:04d}'
        score = similarity[row, col]
        writer.writerow([query, reference, score])
print('Done')
