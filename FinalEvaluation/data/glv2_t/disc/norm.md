# MTT3

| **Metodas**                       | **$\mu$ AP**  | **R@P=90\%** | **R@Rank1**  | **R@Rank10**  |
|-----------------------------------|---------------|--------------|--------------|---------------|
| GIST                              | 0,088 \%      | 0,007 \%     | 0,264 \%     | 0,885 \%      |
| MultiGrain                        | 8,493 \%      | 2,786 \%     | 2,675 \%     | 11,640 \%     |
| **HOW**                           | **19,189 \%** | **9,532 \%** | **4,274 \%** | **20,591 \%** |
| **VisionForce**                   | **12,807 \%** | **6,811 \%** | **3,428 \%** | **14,639 \%** |
| Kosinuso panašumas (ResNetV2-101) | 0 \%          | -            | 0 \%         | 0 \%          |
| Kosinuso panašumas (InceptionV3)  | 0 \%          | -            | 0 \%         | 0 \%          |


## ViT-LQ with normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 01            | 0.02913               | 0.00126           | 0.0598449             | 0.02041              | 0.07610               |
| 02            | 0.02646               | 0.00271           | 0.0602024             | 0.01757              | 0.06870               |
| 03            | 0.01294               | 0.00013           | 0.0807733             | 0.01579              | 0.05456               |
| 04            | 0.01336               | 0.00020           | 0.10196               | 0.01579              | 0.05853               |
| 05            | 0.02001               | 0.00185           | 0.078573              | 0.01625              | 0.05780               |
| 06            | 0.01058               | 0.00026           | 0.0747038             | 0.01407              | 0.05034               |
| 07            | 0.01298               | -                 | -                     | 0.01671              | 0.05919               |
| 08            | 0.01095               | -                 | -                     | 0.01685              | 0.05985               |
| 09            | 0.01338               | 0.00007           | 0.0945321             | 0.01552              | 0.05219               |
| 10            | 0.01211               | 0.00059           | 0.105404              | 0.01506              | 0.05166               |
| 11            | 0.01706               | 0.00013           | 0.109019              | 0.01678              | 0.05859               |
| 12            | 0.01524               | -                 | -                     | 0.01698              | 0.06084               |
| 13            | 0.01863               | -                 | -                     | 0.01592              | 0.05972               |
| 14            | 0.02094               | 0.00040           | 0.114065              | 0.01645              | 0.06295               |
| 15            | 0.02146               | 0.00165           | 0.0982456             | 0.01632              | 0.05952               |
| 16            | 0.01695               | 0.00026           | 0.126434              | 0.01533              | 0.05912               |
| 17            | 0.02420               | 0.00007           | 0.119211              | 0.01731              | 0.06388               |
| 18            | 0.01870               | 0.00026           | 0.134669              | 0.01671              | 0.05959               |