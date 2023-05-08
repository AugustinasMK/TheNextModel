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
| 01            | 0.02057               | 0.00079           | 0.0442313             | 0.01810              | 0.06804               |
| 02            | 0.01630               | 0.00258           | 0.0436757             | 0.01539              | 0.05265               |
| 03            | 0.01647               | 0.00178           | 0.0629334             | 0.01579              | 0.05390               |
| 04            | 0.01598               | 0.00139           | 0.0591423             | 0.01559              | 0.05311               |
| 05            | 0.01246               | 0.00020           | 0.0688357             | 0.01467              | 0.05192               |
| 06            | 0.01211               | 0.00132           | 0.06371               | 0.01447              | 0.05054               |
| 07            | 0.01306               | -                 | -                     | 0.01414              | 0.05047               |
| 08            | 0.01428               | -                 | -                     | 0.01533              | 0.05106               |
| 09            | 0.01645               | 0.00185           | 0.0670318             | 0.01400              | 0.04915               |
| 10            | 0.01243               | 0.00059           | 0.0729364             | 0.01348              | 0.04591               |
| 11            | 0.11961               | 0.02008           | 0.0663933             | 0.03323              | 0.15848               |
| 12            | 0.12862               | 0.03125           | 0.0723431             | 0.03435              | 0.16581               |
| 13            | 0.12842               | 0.03382           | 0.0782522             | 0.03455              | 0.16495               |
| 14            | 0.13517               | 0.03389           | 0.0807115             | 0.03435              | 0.16720               |
| 15            | 0.13261               | 0.03290           | 0.0847093             | 0.03415              | 0.16759               |
| 16            | 0.13755               | 0.03481           | 0.0842384             | 0.03514              | 0.16766               |
| 17            | 0.13335               | 0.03699           | 0.0864771             | 0.03495              | 0.16660               |
| 18            | 0.13276               | 0.03197           | 0.0894803             | 0.03376              | 0.16264               |
| 19            | 0.13707               | 0.03746           | 0.0928542             | 0.03349              | 0.16402               |
| 20            | 0.13894               | 0.03627           | 0.0973553             | 0.03336              | 0.16885               |