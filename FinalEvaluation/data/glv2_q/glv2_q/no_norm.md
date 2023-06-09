# MTT3

| **Metodas**                       | **$\mu$ AP**  | **R@P=90\%** | **R@Rank1**  | **R@Rank10**  |
|-----------------------------------|---------------|--------------|--------------|---------------|
| GIST                              | 0,088 \%      | 0,007 \%     | 0,264 \%     | 0,885 \%      |
| MultiGrain                        | 8,493 \%      | 2,786 \%     | 2,675 \%     | 11,640 \%     |
| **HOW**                           | **19,189 \%** | **9,532 \%** | **4,274 \%** | **20,591 \%** |
| **VisionForce**                   | **12,807 \%** | **6,811 \%** | **3,428 \%** | **14,639 \%** |
| Kosinuso panašumas (ResNetV2-101) | 0 \%          | -            | 0 \%         | 0 \%          |
| Kosinuso panašumas (InceptionV3)  | 0 \%          | -            | 0 \%         | 0 \%          |


## ViT-LQ without normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 01            | 0.02609               | 0.00832           | 0.976916             | 0.01810              | 0.06738               |
| 02            | 0.02089               | 0.00469           | 0.975261             | 0.01539              | 0.05258               |
| 03            | 0.02129               | 0.00548           | 0.966536             | 0.01579              | 0.05377               |
| 04            | 0.02097               | 0.00628           | 0.965723             | 0.01559              | 0.05311               |
| 05            | 0.01944               | 0.00476           | 0.967333             | 0.01467              | 0.05186               |
| 06            | 0.01718               | 0.00502           | 0.963699             | 0.01447              | 0.05054               |
| 07            | 0.01841               | 0.00495           | 0.962676             | 0.01414              | 0.05047               |
| 08            | 0.01951               | 0.00423           | 0.963501             | 0.01533              | 0.05106               |
| 09            | 0.01751               | 0.00436           | 0.959601             | 0.01400              | 0.04915               |
| 10            | 0.01543               | 0.00416           | 0.960889             | 0.01348              | 0.04584               |
| 11            | 0.13087               | 0.04380           | 0.950345             | 0.03323              | 0.15848               |
| 12            | 0.13759               | 0.04736           | 0.942171             | 0.03435              | 0.16581               |
| 13            | 0.13941               | 0.04822           | 0.936384             | 0.03455              | 0.16495               |
| 14            | 0.14621               | 0.04525           | 0.936501             | 0.03435              | 0.16720               |
| 15            | 0.14398               | 0.04783           | 0.93304              | 0.03415              | 0.16759               |
| 16            | 0.14427               | 0.05139           | 0.928526             | 0.03514              | 0.16766               |
| 17            | 0.14348               | 0.04697           | 0.929269             | 0.03495              | 0.16660               |
| 18            | 0.13451               | 0.04168           | 0.926985             | 0.03376              | 0.16264               |
| 19            | 0.14087               | 0.04822           | 0.922493             | 0.03349              | 0.16402               |
| 20            | 0.14252               | 0.04347           | 0.924328             | 0.03336              | 0.16885               |
| 21            | 0.14084               | 0.04512           | 0.920139             | 0.03442              | 0.16819               |
| 22            | 0.14408               | 0.04875           | 0.915534             | 0.03481              | 0.17017               |
| 23            | 0.13536               | 0.04162           | 0.918103             | 0.03277              | 0.16323               |
| 24            | 0.13741               | 0.04049           | 0.919283             | 0.03409              | 0.16627               |
| 25            | 0.13410               | 0.03944           | 0.916211             | 0.03349              | 0.16211               |


