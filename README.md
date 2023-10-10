# METO-S2S
This is a implementation of METO-S2S. The implementations of METO-S2S: A S2S based vessel trajectory prediction method with
Multiple-semantic Encoder and Type-Oriented Decoder.

# Requirements
Python 3.9
cd METO-S2S
pip install requirements.txt

# Data Processing

*.csv --->  *.json

```
cd preprocess
python traj_preprocessing.py
```

## Model Running: train and test
Our trained model is listed in ./models/seq2seq.pkl.
```
cd algorithm
python seq2seq.py
```

## Citation
```

```
