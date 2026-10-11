| Model                     | Backbone        | Classifier   |   Test Acc (%) |   Macro Prec (%) |   Macro F1 (%) |   Macro Rec (%) |   Error Rate (%) |
|:--------------------------|:----------------|:-------------|---------------:|-----------------:|---------------:|----------------:|-----------------:|
| Custom CNN + SVM          | Custom CNN      | SVM          |          97.32 |            96.83 |          96.75 |           96.69 |             2.68 |
| MobileNetV2 + SVM         | MobileNetV2     | SVM          |          97    |            96.73 |          96.27 |           95.88 |             3    |
| Custom CNN + KNN          | Custom CNN      | KNN          |          96.93 |            96.81 |          96.65 |           96.56 |             3.07 |
| EfficientNet-B0 + SVM     | EfficientNet-B0 | SVM          |          96.61 |            95.86 |          95.92 |           96.06 |             3.39 |
| EfficientNet-B0 + KNN     | EfficientNet-B0 | KNN          |          96.22 |            95.49 |          95.52 |           95.65 |             3.78 |
| Custom CNN + RF           | Custom CNN      | RF           |          96.06 |            95.87 |          95.41 |           95.05 |             3.94 |
| MobileNetV2 + KNN         | MobileNetV2     | KNN          |          96.03 |            95.84 |          95.46 |           95.25 |             3.97 |
| MobileNetV2 + RF          | MobileNetV2     | RF           |          95.8  |            96.03 |          95.08 |           94.34 |             4.2  |
| EfficientNet-B0 + RF      | EfficientNet-B0 | RF           |          95.8  |            95.17 |          94.93 |           94.76 |             4.2  |
| EfficientNet-B0 + Softmax | EfficientNet-B0 | Softmax      |          95.7  |            94.7  |          94.94 |           95.38 |             4.3  |
| MobileNetV2 + Softmax     | MobileNetV2     | Softmax      |          94.57 |            94.8  |          93.57 |           92.9  |             5.43 |
| EfficientNet-B0 + NB      | EfficientNet-B0 | NB           |          93.99 |            92.44 |          92.38 |           92.41 |             6.01 |
| MobileNetV2 + NB          | MobileNetV2     | NB           |          93.7  |            92.59 |          92.31 |           92.08 |             6.3  |
| Custom CNN + NB           | Custom CNN      | NB           |          68.44 |            71.74 |          64.79 |           71.84 |            31.56 |
| Custom CNN + Softmax      | Custom CNN      | Softmax      |          68.12 |            76.73 |          63.03 |           66.76 |            31.88 |