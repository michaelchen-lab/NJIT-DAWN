# Deep Ad-Block Whitelist Network (DAWN)

The fast increase in ad-blocker usage has resulted in significant revenue loss for online publishers. To mitigate this, many publishers implement the Wall strategy, where an ad-blocking user is asked to whitelist the intended webpage. If the user refuses, the result is a loss-loss situation: the user is denied access to content, and the publisher cannot receive revenue. An alternative strategy, called AAX, is to show only acceptable ads to users. However, acceptable ads generate less revenue than regular ads.

DAWN is a novel deep learning-based whitelist prediction model, to help implement a personalized counter-ad-blocking policy that dynamically chooses a counter-ad-blocking strategy for individual users. It captures page characteristics, user interests in pages and their sensitivity to ads, reflected in historic behavior, using a deep learning mechanism.

DAWN is built on top of DeepCTR ([https://github.com/shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR)).

## Research Paper

(add link to research and our paper's citation here)

## Dataset

Our research is conducted in collaboration with Forbes Media. Due to the confidential nature of the dataset, it cannot be made public.

However, a sample fake dataset is provided in `dataset.csv`.

## How to Run

1. Install library requirements
```
pip install requirements.txt
```

2. Train model
```
python train.py
```



