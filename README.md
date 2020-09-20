# MatchMaker: A Deep Learning Framework for Drug Synergy Prediction

---
## Abstract
Drug combination therapies have been a viable strategy for the treatment of complex diseases such as cancer due to increased efficacy and reduced side effects. However, experimentally validating all possible combinations for synergistic interaction even with highthroughout screens is intractable due to vast combinatorial search space. Computational techniques can reduce the number of combinations to be evaluated experimentally by prioritizing promising candidates. We present MatchMaker that predicts drug synergy scores using drug chemical structure information and gene expression profiles of cell lines in a deep learning framework. For the first time, our model utilizes the largest known drug combination dataset to date, DrugComb. We compare the performance of MatchMaker with the state-of-the-art models and observe up to ∼ 20% correlation and ∼ 40% mean squared error (MSE) improvements over the next best method. We investigate the cell types and drug pairs that are relatively harder to predict and present novel candidate pairs.

---

## Authors
Halil Ibrahim Kuru, Oznur Tastan, A. Ercument Cicek
Our paper is available at <a href="https://www.biorxiv.org/content/10.1101/2020.05.24.113241v3">**bioRxiv**</a>

---

## Instructions Manual

### Requirements
- Python 3.7
- Numpy 1.18.1 
- Scipy 1.4.1
- Pandas 1.0.1
- Tensorflow 2.1.0
- Tensorflow-gpu 2.1.0
- Scikit-Learn 0.22.1
- keras-metrics 1.1.0
- h5py 2.10.0
- cudnn 7.6.5 (for gpu support only)

### Data
Raw data of drug combinations are taken from <a href="https://drugcomb.fimm.fi/">**DrugComb**</a>

Drug chemical features are calculated by Chemopy

RMA normalized E-MTAB-3610 untrated cell line gene expression data is downloaded from <a href="https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Home_files/Extended%20Methods.html#8">**cancerrxgene**</a>

You can download preprocessed data from <a href="https://drive.google.com/open?id=1eQpwJKiIdMI0wTz_GEa285q0GHUr6wRe">**link**</a>, extract all files into `data/`

### Training
```shell
$ python main.py --saved-model-name matchmaker.h5 --train-test-mode 1
```

### Testing with pretrained Model
Download pretrained weights from <a href="https://drive.google.com/open?id=1QtMw0unMI-ZY-0z6_1bF76Cf627zXDWz">**link**</a>
```shell
$ python main.py --saved-model-name matchmaker_saved.h5 --train-test-mode 0
```

### NEWS:
We build a command line application <a href="https://github.com/hikuru/MatchMakerApp">**MatchMakerApp**</a>. You can play and predict synergy of any two drugs by using their PubChem CIDs.

---

## References
- Zagidullin, B., Aldahdooh, J., Zheng, S., Wang, W., Wang, Y., Saad, J., ... & Tang, J. (2019). DrugComb: an integrative cancer drug combination data portal. Nucleic acids research, 47(W1), W43-W51.
- CCao, D. S., Xu, Q. S., Hu, Q. N., & Liang, Y. Z. (2013). ChemoPy: freely available python package for computational biology and chemoinformatics. Bioinformatics, 29(8), 1092-1094.
- [dataset] Francesco Iorio (2015). Transcriptional Profiling of 1,000 human cancer cell lines, arrayexpress-repository, V1. https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-3610.


## License

- **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**
