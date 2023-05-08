# MTT3

| **Metodas**                       | **μAP**       | **R@P=90\%**  | **R@Rank1**   | **R@Rank10**  |
|-----------------------------------|---------------|---------------|---------------|---------------|
| GIST                              | 8,824 \%      | 6,435 \%      | 18,250 \%     | 20,261 \%     |
| **MultiGrain**                    | **34,603 \%** | **27,652 \%** | **42,333 \%** | **49,623 \%** |
| **HOW**                           | **37,977 \%** | **26,345 \%** | **46,154 \%** | **48,819 \%** |
| **VisionForce**                   | **85,784 \%** | **77,526 \%** | **90,900 \%** | **93,012 \%** |
| Kosinuso panašumas (ResNetV2-101) | 2,823 \%      | 1,810 \%      | 23,379 \%     | 27,602 \%     |
| Kosinuso panašumas (InceptionV3)  | 3,524 \%      | 1,860 \%      | 21,317 \%     | 25,842 \%     |


## ViT-D with normalizations

| **Iteration** | **Average Precision** | **Recall at P90** | **Threshold at P90** | **Recall at rank 1** | **Recall at rank 10** |
|---------------|-----------------------|-------------------|----------------------|----------------------|-----------------------|
| 01            | 0.03385               | -                 | -                    | 0.42484              | 0.46204               |
| 02            | 0.04142               | -                 | -                    | 0.41981              | 0.44847               |
| 03            | 0.04409               | 0.00151           | 0.13606              | 0.42232              | 0.45500               |
| 04            | 0.03354               | -                 | -                    | 0.43992              | 0.46456               |
| 05            | 0.05052               | 0.00855           | 0.104766             | 0.45148              | 0.47964               |
| 06            | 0.04398               | 0.00654           | 0.133106             | 0.41176              | 0.43942               |
| 07            | 0.03958               | 0.00452           | 0.120164             | 0.42383              | 0.45852               |
| 08            | 0.03588               | 0.00050           | 0.148252             | 0.44042              | 0.47612               |
| 09            | 0.04304               | 0.00050           | 0.139371             | 0.43338              | 0.46858               |
| 10            | 0.01740               | -                 | -                    | 0.43389              | 0.47260               |
| 11            | 0.13641               | -                 | -                    | 0.50126              | 0.54701               |
| 12            | 0.13677               | 0.00101           | 0.16472              | 0.50779              | 0.55757               |
| 13            | 0.24067               | 0.12066           | 0.12488              | 0.51433              | 0.56812               |
| 14            | 0.18475               | 0.02262           | 0.146981             | 0.53243              | 0.58874               |
| 15            | 0.17066               | -                 | -                    | 0.51584              | 0.57014               |
| 16            | 0.15064               | 0.00101           | 0.177051             | 0.50628              | 0.56662               |
| 17            | 0.20236               | 0.07491           | 0.142308             | 0.49774              | 0.56511               |
| 18            | 0.09652               | 0.00050           | 0.210275             | 0.50427              | 0.55354               |
| 19            | 0.14654               | 0.00201           | 0.183938             | 0.48316              | 0.54651               |
| 20            | 0.12768               | 0.00201           | 0.176085             | 0.49472              | 0.54952               |
| 21            | 0.20291               | 0.02212           | 0.168892             | 0.50427              | 0.56461               |
| 22            | 0.14872               | 0.00101           | 0.181753             | 0.50679              | 0.56159               |
| 23            | 0.18841               | 0.02313           | 0.166628             | 0.50729              | 0.56762               |
| 24            | 0.19418               | 0.05983           | 0.155286             | 0.49874              | 0.56561               |
| 25            | 0.21689               | 0.09050           | 0.149265             | 0.48819              | 0.55254               |