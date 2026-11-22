# Crop Recommendation Engine / crop

Selected model: **RandomForest**  |  split: stratified random 80/20  |  rows: 2,200

| metric | reference | delivered |
|---|---|---|
| accuracy | 0.9386 | 0.9477 |
| macro_precision | 0.9427 | 0.9507 |
| macro_recall | 0.9386 | 0.9477 |
| macro_f1 | 0.9392 | 0.9477 |
| weighted_f1 | 0.9392 | 0.9477 |

## Per class

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| apple | 1.000 | 1.000 | 1.000 | 20 |
| banana | 1.000 | 1.000 | 1.000 | 20 |
| blackgram | 1.000 | 0.900 | 0.947 | 20 |
| chickpea | 0.952 | 1.000 | 0.976 | 20 |
| coconut | 1.000 | 1.000 | 1.000 | 20 |
| coffee | 1.000 | 1.000 | 1.000 | 20 |
| cotton | 0.952 | 1.000 | 0.976 | 20 |
| grapes | 1.000 | 1.000 | 1.000 | 20 |
| jute | 0.909 | 1.000 | 0.952 | 20 |
| kidneybeans | 1.000 | 0.950 | 0.974 | 20 |
| lentil | 1.000 | 0.950 | 0.974 | 20 |
| maize | 0.789 | 0.750 | 0.769 | 20 |
| mango | 1.000 | 1.000 | 1.000 | 20 |
| mothbeans | 0.792 | 0.950 | 0.864 | 20 |
| mungbean | 1.000 | 1.000 | 1.000 | 20 |
| muskmelon | 1.000 | 1.000 | 1.000 | 20 |
| orange | 0.708 | 0.850 | 0.773 | 20 |
| papaya | 1.000 | 1.000 | 1.000 | 20 |
| pigeonpeas | 1.000 | 0.950 | 0.974 | 20 |
| pomegranate | 0.812 | 0.650 | 0.722 | 20 |
| rice | 1.000 | 0.900 | 0.947 | 20 |
| watermelon | 1.000 | 1.000 | 1.000 | 20 |

## Most frequent confusions (true -> predicted)

- pomegranate -> orange: 7
- maize -> mothbeans: 4
- orange -> pomegranate: 3
- rice -> jute: 2
- blackgram -> maize: 2