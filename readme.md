# ZeroShot-AnomalyDist

## Project Summary

This project systematically benchmarks several zero-shot probabilistic methods for anomaly detection on the SECOM industrial dataset. In addition to classic supervised models (CNN, XGBoost), a range of heuristic optimization algorithms—including brute-force search, simulated annealing, and particle swarm optimization (PSO)—were used to tune classifier hyperparameters. The main focus, however, is on comparing standard statistical distributions (Gaussian, multivariate t, Laplace, Cauchy) as the basis for unsupervised zero-shot anomaly scoring, with no manual parameter tuning for these methods. Performance is compared with classic supervised baselines, providing a reference for future work.

本專案針對 SECOM 工業數據集，系統性比較多種「Zero-shot 分布法」進行異常檢測。除了監督式模型（CNN、XGBoost）外，亦實作了暴力搜尋、模擬退火（simulated annealing）、粒子群優化（PSO）等啟發式演算法來進行參數自動化調整。然而所有分布法（Zero-shot）均不進行人工調參，直接以資料中正常樣本統計特性為依據，建立異常分數（anomaly score）。傳統監督式模型則作為參考基線。

---

## Contents

**Supervised baseline models:**

- 1D CNN
- XGBoost Classifier

**Heuristic optimization for supervised models:**

- Brute-force search
- Simulated annealing
- Particle swarm optimization (PSO)

**Zero-shot distributional anomaly detectors:**

- Gaussian (Multivariate Normal)
- Multivariate t-distribution
- Laplace (per dimension, independent)
- Cauchy (per dimension, independent)

**Evaluation metrics:**

- Precision, Recall, F1-score, Confusion Matrix
- AUROC, AUPRC

---

## Main Findings

| Method         | Precision (Class 1) | Recall (Class 1) | F1-score (Class 1) |
| -------------- | ------------------- | ---------------- | ------------------ |
| CNN            | 0.00                | 0.00             | 0.00               |
| XGBoost        | 0.00 – 0.16         | 0.00 – 0.16      | 0.00 – 0.16        |
| Laplace        | **0.31**            | **0.24**         | **0.27**           |
| Gaussian       | 0.25                | 0.19             | 0.22               |
| t-distribution | 0.25                | 0.19             | 0.22               |
| Cauchy         | 0.13                | 0.10             | 0.11               |

> Laplace distribution outperformed other zero-shot statistical methods for anomaly detection in this dataset.

---

## Usage

1. Download and extract the [SECOM dataset](https://archive.ics.uci.edu/dataset/229/secom) from UCI.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `ZeroShot-AnomalyDist.ipynb` for complete workflow and reproducible results.

---

## Notes

- No feature engineering, cross-validation, or manual parameter tuning was performed for any zero-shot methods.
- All steps are reproducible with the included notebook and standard Python packages (pandas, numpy, scikit-learn, scipy, matplotlib, xgboost, torch).
- The notebook is fully annotated in Traditional Chinese.

---

## Author

洪禎\
Graduate Institute of Telecommunications and Communication, National Chung Cheng University\
[GitHub: scuranger0625](https://github.com/scuranger0625)

---

## License

MIT License

