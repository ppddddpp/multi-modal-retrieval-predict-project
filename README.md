# Unified Multimodal Framework for Chest X-Ray Retrieval and Disease Prediction

This repository contains the official implementation of **"A Unified Multimodal Framework for Chest X-Ray Retrieval and Disease Prediction: Towards Interpretable and Reproducible Medical AI."**  
It was developed as part of the HCMUS final project and integrates multimodal learning, retrieval, and explainability for medical imaging.

---

## Overview

### Features

* Disease-aware joint embeddings from image and text (DICOM + report)
* Knowledge-graph integration for semantic consistency and reranking
* Multimodal disease prediction and retrieval
* Attention-based and Integrated Gradient (IG) explanations
* Flask web interface for interactive demo and debugging

---

## Project Structure

```
project_root/
├── data/               # Raw input (DICOM, XML)
├── knowledge_graph/    # KG triples, embeddings, node2id/relation2id
├── embeddings/         # Generated embedding files (.npy, .json)
├── checkpoints/        # Model weights
├── outputs/            # Labeled CSVs and evaluation logs
├── ground_truths/      # JSONs with test-to-train relevance
├── splited_data/       # Split datasets
├── models/             # Swin, BERT, Spacy, etc.
├── src/                # Core Python code
└── config/             # Training/eval config YAMLs
```

---

## Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

### 2. Data Preparation

* Place XML reports under `data/openi/xml/`
* Place DICOMs under `data/openi/dicom/`
* Overview of data folder structure can be found at dataFolderStructure.txt

### 3. Labeling and Splitting

```bash
python src/data_run.py
```

Generates `outputs/openi_labels_final.csv`.  
Split datasets are stored in `splited_data/`.

## 4. Train Knowledge Graph and Model

```bash
python src/Trainner/train.py
```

Trained CompGCN/TransE embeddings saved in `knowledge_graph/` and trained model saved in `checkpoints/`


### 5. Retrieval Evaluation

```bash
python src/Helpers/contructGT.py
```

Generates `ground_truths/test_relevance.json`. Then run evaluate in `src/Evaluate/`
Metrics include:
* Precision@K, Recall@K, nDCG, mAP  
* Per-class AUROC/F1 table  
* Statistical significance testing for ablation gains  

### 6. Web Demo

```bash
python src/web/app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore the interface.

---

## Dependencies

* PyTorch
* HuggingFace Transformers
* TimM (Swin Transformer)
* SciSpaCy (`en_core_sci_sm`)
* Flask
* Captum (for explainability)
* scikit-learn, numpy, pandas, networkx, rdflib
* Additional can be found in requirement.txt
---

## Evaluation

Retrieval relevance is based on label overlap and ontology-aware relationships.  
Evaluation reports include:
* Per-class metrics table (AUROC, F1)
* Confusion and calibration plots
* Statistical significance for ablation gains (paired t-test)

---

## Example Demos

### Predict & Explain
![Predict & Explain](demo/demo_1.png)
![Predict & Explain](demo/demo_2.png)
![Predict & Explain](demo/demo_3.png)

### Retrieval Demo
![Retrieval Demo](demo/demo_4.png)
![Predict & Explain](demo/demo_5.png)
![Predict & Explain](demo/demo_6.png)
---

## Dataset

* **NIH OpenI Dataset** — Chest X-rays and associated reports.  
  [https://openi.nlm.nih.gov](https://openi.nlm.nih.gov)

---

## Trained Models & Embeddings

Pretrained models and embeddings are available at:  
[https://huggingface.co/ppddddpp/unified-multimodal-chestxray](https://huggingface.co/ppddddpp/unified-multimodal-chestxray)

---

## Acknowledgments

* [OpenI Dataset](https://openi.nlm.nih.gov/)
* [Disease Ontology (DOID)](http://purl.obolibrary.org/obo/doid.obo)
* [RadLex Ontology](https://bioportal.bioontology.org/ontologies/RADLEX)
* ClinicalBERT / Swin Transformer / SciSpaCy / Captum

---

## Reproducibility

All source code, pretrained models, and configuration files are publicly available to ensure full reproducibility.  
Scripts for data preprocessing, training, retrieval evaluation, and visualization are included under `src/`.
