# Data

Not tracked in git (see `.gitignore`). Download the Kaggle "Garbage Classification" dataset and place it here so the layout looks like:

```
data/
  raw/
    cardboard/*.jpg
    glass/*.jpg
    metal/*.jpg
    paper/*.jpg
    plastic/*.jpg
    trash/*.jpg
```

`src/data/dataset.py` uses `torchvision.datasets.ImageFolder`, which infers class labels directly from these subfolder names (sorted alphabetically) — there is no separate class list to keep in sync.

On Kaggle, add the dataset via "Add Input" in the notebook UI instead of downloading it here; `notebooks/kaggle_train.ipynb` points `DATA_DIR` at `/kaggle/input/...` accordingly.
