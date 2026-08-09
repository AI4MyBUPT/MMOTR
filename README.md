# MCSP-OTMR

Official implementation of **MCSP-OTMR: Multimodal Cancer Survival Prediction with Optimal Transport-based Missing Modality Reconstruction**.

<p align="center">
  <img src="assets/Figure_MCSP_OTMR.jpg" width="1500px" />
</p>

## Introduction

MCSP-OTMR integrates whole-slide image features and functional genomic signatures for multimodal cancer survival prediction. The framework supports both complete-modality inference and missing-genomic-modality inference.

The method contains five main components:

- **Feature Process:** WSI patch features and six functional genomic groups are encoded into compact embeddings.
- **Compression and Diversity (DVIB):** pathology features are compressed into diverse latent pathology components.
- **Missing Modality Reconstruction (OTCR):** optimal transport aligns pathology components with genomic components and guides missing genomic reconstruction.
- **Feature Fusion and Prediction:** cross-attention fuses pathology and genomics, followed by modality-specific Transformer branches for survival risk prediction.
- **Fisher Metric and Reweighting (FIMR):** Fisher information is used to monitor modality imbalance and reweight gradients during training.

## Data Preparation

### WSIs

1. Download the original WSI data from [TCGA](https://portal.gdc.cancer.gov/).
2. Preprocess WSIs and extract patch-level features with [CLAM](https://github.com/mahmoodlab/CLAM), CTransPath, or another compatible feature extractor.

The expected feature directory structure is:

```text
DATA_ROOT_DIR/
    pt_files/
        slide_1.pt
        slide_2.pt
        ...
```

### Genomics

The cleaned survival metadata and genomic signature files are provided under:

```text
dataset_csv/
datasets_csv_sig/
splits/
```

## Requirements

```bash
conda create -n mcsp-otmr python=3.9
conda activate mcsp-otmr
pip install -r requirements.txt
```

The OTCR module depends on the Python Optimal Transport package, imported as `ot`.

## Usage

The current main training entry is:

```bash
CUDA_VISIBLE_DEVICES=<CUDA_IDX> python main_fl.py \
  --split_dir <TCGA_DATASET> \
  --data_root_dir <WSI_FEATURE_DIR>
```

For example, `<TCGA_DATASET>` can be `tcga_brca`, `tcga_gbmlgg`, `tcga_luad`, or `tcga_ucec`, matching the folders under `splits/5foldcv/`.

## Acknowledgement

This implementation builds on commonly used multimodal survival prediction components and feature preprocessing tools. We thank the following repositories:

- [CLAM](https://github.com/mahmoodlab/CLAM)
- [CTransPath](https://github.com/Xiyue-Wang/TransPath)
- [MCAT](https://github.com/mahmoodlab/MCAT)
