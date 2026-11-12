# Data

Raw data files are **not** committed to this repository (sensor files are large and redistribution-restricted).

## SisFall

1. Download from: http://sistemic.udea.edu.co/en/research/projects/english-falls/
2. Extract so the structure is:
   ```
   data/raw/sisfall/
   ├── SA01/   (Subject Adult 01 — young adults)
   ├── ...
   ├── SE01/   (Subject Elderly 01)
   ├── ...
   └── SisFall_dataset.xlsx   (optional metadata)
   ```
   Each subject folder contains `.txt` files named e.g. `F01SA01R01.txt` (fall type, subject, repetition).

## HAR70+ / Tanwar Kaggle

1. Download from: https://www.kaggle.com/datasets/tanwarelectrical/activity-recognition-in-senior-citizens
2. Place CSVs in `data/raw/har70plus/`

## Laavanya Kaggle

1. Download from: https://www.kaggle.com/datasets/laavanya/elderly-fall-prediction-and-detection
2. Place files in `data/raw/laavanya/`

## Automated download

```bash
python scripts/download_data.py --dataset sisfall   # requires Kaggle credentials for the Kaggle sets
```
