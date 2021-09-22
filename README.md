## Covid Challenge
Code for the covid challenge.


## To run

Create a new virtual environment

```bash
conda create -n covid python=3.7
conda activate covid
pip install -r requirements.txt
```

## Preprocess
Gives out embeddings (mfcc, mel_spectogram, ege_maps, wav2vec2)
```python
python get_embeddings.py
```

## Model
Work in progress