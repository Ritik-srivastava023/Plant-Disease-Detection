| Model                     | Backbone        | Head    |   Test Acc (%) |   Error Rate (%) |   MMRE |   SD_MRE |   MdMRE |   PRED(0.25) (%) |   PRED(0.10) (%) |   Class MMRE (%) |
|:--------------------------|:----------------|:--------|---------------:|-----------------:|-------:|---------:|--------:|-----------------:|-----------------:|-----------------:|
| Custom CNN + SVM          | Custom CNN      | SVM     |          97.32 |             2.68 | 0.0498 |   0.1487 |  0.0033 |            94.12 |            89.47 |             1.14 |
| MobileNetV2 + SVM         | MobileNetV2     | SVM     |          97    |             3    | 0.0513 |   0.1546 |  0.0026 |            93.83 |            89.7  |             2.67 |
| Custom CNN + KNN          | Custom CNN      | KNN     |          96.93 |             3.07 | 0.045  |   0.1517 |  0      |            94.12 |            88.73 |             2.27 |
| EfficientNet-B0 + SVM     | EfficientNet-B0 | SVM     |          96.61 |             3.39 | 0.0533 |   0.1611 |  0.0023 |            93.64 |            89.44 |             2.7  |
| EfficientNet-B0 + KNN     | EfficientNet-B0 | KNN     |          96.22 |             3.78 | 0.0549 |   0.1703 |  0      |            92.86 |            86.98 |             3.32 |
| Custom CNN + RF           | Custom CNN      | RF      |          96.06 |             3.94 | 0.1739 |   0.2056 |  0.0913 |            73.93 |            51.94 |             3.09 |
| MobileNetV2 + KNN         | MobileNetV2     | KNN     |          96.03 |             3.97 | 0.06   |   0.1732 |  0      |            92.18 |            85.24 |             4.01 |
| MobileNetV2 + RF          | MobileNetV2     | RF      |          95.8  |             4.2  | 0.207  |   0.2108 |  0.1342 |            67.38 |            43.44 |             4.04 |
| EfficientNet-B0 + RF      | EfficientNet-B0 | RF      |          95.8  |             4.2  | 0.1926 |   0.2123 |  0.1134 |            71.38 |            47.13 |             2.34 |
| EfficientNet-B0 + Softmax | EfficientNet-B0 | Softmax |          95.7  |             4.3  | 0.0584 |   0.1802 |  0.0003 |            92.57 |            88.79 |             4.62 |
| MobileNetV2 + Softmax     | MobileNetV2     | Softmax |          94.57 |             5.43 | 0.0667 |   0.1971 |  0.0002 |            91.7  |            87.86 |             6.84 |
| EfficientNet-B0 + NB      | EfficientNet-B0 | NB      |          93.99 |             6.01 | 0.0599 |   0.2347 |  0      |            93.83 |            93.73 |             3.02 |
| MobileNetV2 + NB          | MobileNetV2     | NB      |          93.7  |             6.3  | 0.063  |   0.2408 |  0      |            93.57 |            93.48 |             2.35 |
| Custom CNN + NB           | Custom CNN      | NB      |          68.44 |            31.56 | 0.3148 |   0.4612 |  0      |            68.15 |            67.89 |           123.37 |
| Custom CNN + Softmax      | Custom CNN      | Softmax |          68.12 |            31.88 | 0.3503 |   0.4144 |  0.081  |            59.3  |            51.61 |            72.27 |