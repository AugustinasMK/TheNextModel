# MTT3

| **Metodas**                       | **μAP**       | **R@P=90\%**  | **R@Rank1**   | **R@Rank10**  |
|-----------------------------------|---------------|---------------|---------------|---------------|
| GIST                              | 8,824 \%      | 6,435 \%      | 18,250 \%     | 20,261 \%     |
| **MultiGrain**                    | **34,603 \%** | **27,652 \%** | **42,333 \%** | **49,623 \%** |
| **HOW**                           | **37,977 \%** | **26,345 \%** | **46,154 \%** | **48,819 \%** |
| **VisionForce**                   | **85,784 \%** | **77,526 \%** | **90,900 \%** | **93,012 \%** |
| Kosinuso panašumas (ResNetV2-101) | 2,823 \%      | 1,810 \%      | 23,379 \%     | 27,602 \%     |
| Kosinuso panašumas (InceptionV3)  | 3,524 \%      | 1,860 \%      | 21,317 \%     | 25,842 \%     |


## ViT-D with no normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 01            | 0.24217               | 0.16189           | 0.98396              | 0.44796              | 0.47813               |
| 02            | 0.22555               | 0.12217           | 0.984143             | 0.45651              | 0.49522               |
| 03            | 0.21671               | 0.12469           | 0.978276             | 0.43992              | 0.47813               |
| 04            | 0.22263               | 0.13072           | 0.978349             | 0.44042              | 0.48617               |
| 05            | 0.19426               | 0.11815           | 0.977757             | 0.40774              | 0.45199               |
| 06            | 0.16501               | 0.10709           | 0.977909             | 0.41931              | 0.46405               |
| 07            | 0.21698               | 0.13122           | 0.976589             | 0.45249              | 0.49522               |
| 08            | 0.22867               | 0.14279           | 0.973195             | 0.43841              | 0.49975               |
| 09            | 0.14728               | 0.05028           | 0.984939             | 0.37054              | 0.41579               |
| 10            | 0.18543               | 0.11865           | 0.97539              | 0.39970              | 0.44746               |
| 11            | 0.15029               | 0.08798           | 0.979758             | 0.41579              | 0.45701               |
| 12            | 0.18942               | 0.11865           | 0.972833             | 0.43439              | 0.48819               |
| 13            | 0.20966               | 0.13776           | 0.965244             | 0.44796              | 0.50327               |
| 14            | 0.16510               | 0.07491           | 0.975682             | 0.42936              | 0.48366               |
| 15            | 0.18857               | 0.12267           | 0.963748             | 0.43288              | 0.48617               |
| 16            | 0.22037               | 0.14580           | 0.960341             | 0.45601              | 0.51433               |
| 17            | 0.17878               | 0.10508           | 0.965892             | 0.41629              | 0.48316               |
| 18            | 0.18029               | 0.09050           | 0.967607             | 0.43238              | 0.49271               |
| 19            | 0.17767               | 0.10106           | 0.963219             | 0.45249              | 0.50830               |
| 20            | 0.17093               | 0.09955           | 0.962113             | 0.42584              | 0.48416               |
| 21            | 0.18783               | 0.10508           | 0.960828             | 0.44193              | 0.50075               |
| 22            | 0.19393               | 0.11111           | 0.955878             | 0.44143              | 0.50729               |
| 23            | 0.21522               | 0.13575           | 0.952324             | 0.44595              | 0.50980               |
| 24            | 0.21431               | 0.11966           | 0.958022             | 0.47210              | 0.52690               |
| 25            | 0.18914               | 0.09351           | 0.962183             | 0.45902              | 0.51433               |

