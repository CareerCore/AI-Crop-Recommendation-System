# Crop Recommendation Engine / crop

Selected model: **HistGradientBoosting**  |  split: stratified random 80/20  |  rows: 2,200

| metric | reference | delivered |
|---|---|---|
| accuracy | 0.9932 | 0.9886 |
| macro_precision | 0.9935 | 0.9893 |
| macro_recall | 0.9932 | 0.9886 |
| macro_f1 | 0.9932 | 0.9886 |
| weighted_f1 | 0.9932 | 0.9886 |

## Per class

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| apple | 1.000 | 1.000 | 1.000 | 20 |
| banana | 1.000 | 1.000 | 1.000 | 20 |
| blackgram | 1.000 | 0.950 | 0.974 | 20 |
| chickpea | 1.000 | 1.000 | 1.000 | 20 |
| coconut | 1.000 | 1.000 | 1.000 | 20 |
| coffee | 0.952 | 1.000 | 0.976 | 20 |
| cotton | 1.000 | 1.000 | 1.000 | 20 |
| grapes | 1.000 | 1.000 | 1.000 | 20 |
| jute | 0.952 | 1.000 | 0.976 | 20 |
| kidneybeans | 1.000 | 1.000 | 1.000 | 20 |
| lentil | 1.000 | 0.900 | 0.947 | 20 |
| maize | 0.950 | 0.950 | 0.950 | 20 |
| mango | 1.000 | 1.000 | 1.000 | 20 |
| mothbeans | 0.909 | 1.000 | 0.952 | 20 |
| mungbean | 1.000 | 1.000 | 1.000 | 20 |
| muskmelon | 1.000 | 1.000 | 1.000 | 20 |
| orange | 1.000 | 1.000 | 1.000 | 20 |
| papaya | 1.000 | 1.000 | 1.000 | 20 |
| pigeonpeas | 1.000 | 1.000 | 1.000 | 20 |
| pomegranate | 1.000 | 1.000 | 1.000 | 20 |
| rice | 1.000 | 0.950 | 0.974 | 20 |
| watermelon | 1.000 | 1.000 | 1.000 | 20 |

## Most frequent confusions (true -> predicted)

- lentil -> mothbeans: 2
- rice -> jute: 1
- maize -> coffee: 1
- blackgram -> maize: 1