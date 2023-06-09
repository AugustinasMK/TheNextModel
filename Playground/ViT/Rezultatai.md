# MTT3

| **Metodas**                       | **μAP**       | **R@P=90\%**  | **R@Rank1**   | **R@Rank10**  |
|-----------------------------------|---------------|---------------|---------------|---------------|
| GIST                              | 8,824 \%      | 6,435 \%      | 18,250 \%     | 20,261 \%     |
| **MultiGrain**                    | **34,603 \%** | **27,652 \%** | **42,333 \%** | **49,623 \%** |
| **HOW**                           | **37,977 \%** | **26,345 \%** | **46,154 \%** | **48,819 \%** |
| **VisionForce**                   | **85,784 \%** | **77,526 \%** | **90,900 \%** | **93,012 \%** |
| Kosinuso panašumas (ResNetV2-101) | 2,823 \%      | 1,810 \%      | 23,379 \%     | 27,602 \%     |
| Kosinuso panašumas (InceptionV3)  | 3,524 \%      | 1,860 \%      | 21,317 \%     | 25,842 \%     |

# No GeM

## ViT-L with normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 0             | 0.03911               | 0.0181            | 0.850955             | 0.27049              | 0.32579               |
| 5             | 0.07796               | 0.04525           | 0.665327             | 0.23077              | 0.27149               |
| 10            | 0.04648               | 0.01408           | 0.713617             | 0.21217              | 0.25088               |
| 11            | 0.07772               | 0.03117           | 0.384725             | 0.24233              | 0.29512               |
| 12            | 0.05167               | 0.01810           | 0.349658             | 0.25339              | 0.30216               |
| 13            | 0.09310               | 0.03419           | 0.175329             | 0.36853              | 0.30216               |
| 15            | 0.01500               | 0.00050           | 0.293863             | 0.33635              | 0.42534               |
| 16            | 0.05099               | 0.01056           | 0.193615             | 0.34741              | 0.40171               |
| 16 (200)      | 0.05085               | 0.01056           | 0.193615             | 0.32378              | 0.35445               |
| 20            | 0.05417               | 0.01156           | 0.198819             | 0.34942              | 0.39266               |
| 21            | 0.12951               | 0.06134           | 0.191536             | 0.40372              | 0.46104               |

## ViT-L without normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 0             | 0.05027               | 0.01810           | 0.89229              | 0.26496              | 0.31574               |
| 5             | 0.02812               | -                 | -                    | 0.22826              | 0.26144               |
| 10            | 0.08622               | 0.03167           | 0.94972              | 0.22373              | 0.27702               |
| 11            | 0.05833               | -                 | -                    | 0.24183              | 0.29563               |
| 12            | 0.07140               | 0.01458           | 0.982676             | 0.25993              | 0.31121               |
| 13            | 0.08626               | -                 | -                    | 0.37557              | 0.43288               |
| 15            | 0.07434               | -                 | -                    | 0.34842              | 0.41679               |
| 16            | 0.05464               | -                 | -                    | 0.35897              | 0.42484               |
| 16 (200)      | 0.05451               | -                 | -                    | 0.34942              | 0.39869               |
| 20            | 0.07888               | -                 | -                    | 0.35797              | 0.42031               |
| 21            | 0.08571               | -                 | -                    | 0.40623              | 0.46657               |


# GeM

## ViT-L with normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 5             | 0.03748               | 0.00452           | 0.105841             | 0.38160              | 0.41277               |


## ViT-L without normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 5             | 0.07435               | -                 | -                    | 0.40523              | 0.45098               |


## ViT-L (GLV2) with normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 5Q            | 0.07198               | -                 | -                    | 0.42031              | 0.47009               |
| 5T            | 0.03727               | -                 | -                    | 0.43389              | 0.46757               |


## ViT-L (GLV2) without normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 5Q            | 0.14385               | 0.01056           | 0.995826             | 0.40975              | 0.46154               |
| 5T            | 0.09916               | -                 | -                    | 0.43942              | 0.48014               |


## Just class

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 2 + norm      | 0.00020               | -                 | -                    | 0.09150              | 0.10357               |
| 2             | 0.00151               | -                 | -                    | 0.16290              | 0.19105               |





