# mGPS Optimization

This repository contains code, data, and documentation for hierarchical machine learning approaches to predict the geographic origin of environmental metagenomic samples. The project builds on the MetaSUB dataset and advances the state-of-the-art in microbiome geolocation, surpassing previous tools such as mGPS.

## Overview

Accurate prediction of sample origin from microbial signatures has important applications in biosurveillance, forensic science, and public health. Previous methods, such as mGPS, relied on hierarchical XGBoost models and achieved high city-level accuracy but limited coordinate precision. This project introduces a robust ensemble learning framework that leverages neural networks, GrowNet, gradient boosting, and transformer-based models to deliver substantial improvements in both classification and coordinate prediction.

## Key Features

- **Hierarchical Prediction:** Models predict continent, city, and precise coordinates in a sequential framework.
- **Ensemble Learning:** Combines XGBoost, LightGBM, CatBoost, TabPFN, neural networks, and GrowNet using meta-models for optimal performance.
- **Error Propagation Analysis:** Introduces a mathematical framework to quantify how misclassifications at higher levels affect coordinate predictions.
- **Advanced Feature Selection:** Utilizes recursive feature elimination (RFE) and SMOTE for class imbalance correction.
- **Scalable Architecture:** Designed to handle larger and more diverse datasets than previous approaches.

## Results

- **Tenfold Reduction in Median Coordinate Error:** Ensemble model achieves a median error of 13.7 km, compared to 137 km for mGPS.
- **High Classification Accuracy:** 95% continent accuracy and 93% city accuracy on the MetaSUB dataset.
- **Robustness to Class Imbalance:** Ensemble approach maintains strong performance even for underrepresented regions.
- **Fine-Scale Localization:** Capable of distinguishing sample origins at the level of neighborhoods or districts within cities.

## Repository Structure

```
binp37/
├── data/                # Raw and processed data files
│   ├── metasub/         # MetaSUB dataset files
│   ├── geopandas/       # Geospatial data for mapping
├── notebooks/           # Jupyter notebooks for analysis and visualization
├── scripts/             # Python scripts for preprocessing, modeling, and feature engineering
├── report/              # LaTeX source for manuscript (abstract, introduction, methods, results, discussion)
│   ├── abstract/
│   ├── introduction/
│   ├── methods/
│   ├── results/
│   ├── discussions/
│   ├── references/
├── .gitignore           # Files and folders excluded from version control
```

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/ChandrashekarCR/mgps_optimization.git
   cd mgps_optimization
   ```

2. **Set up the environment:**
   - Install dependencies using `conda` or `pip` as specified in your environment setup.
   - Example:
     ```sh
     conda env create -f environment.yml
     conda activate binp37_env
     ```

3. **Data Preparation:**
   - Download and preprocess the MetaSUB dataset as described in `/scripts/data_preprocess/preprocess_metasub.py`.
   - Filter and select features using RFE and SMOTE.

4. **Model Training:**
   - Train separate neural networks, combined neural networks, GrowNet, and ensemble models using scripts and notebooks provided.
   - Hyperparameter optimization is performed using Optuna.

5. **Evaluation:**
   - Evaluate models using classification accuracy, F1-score, geodesic error, in-radius accuracy, AUC, and AUPR.
   - Compare results with previous state-of-the-art (mGPS).

## Documentation

- **Abstract:** High-level summary of the project and results.
- **Introduction:** Background, motivation, and limitations of previous work.
- **Methods:** Detailed description of dataset, preprocessing, model architectures, and training protocols.
- **Results:** Comprehensive evaluation of all models and comparison with mGPS.
- **Discussion:** Interpretation of findings, comparison with previous work, limitations, and future directions.

## References

- [MetaSUB Project](https://www.sciencedirect.com/science/article/pii/S0092867421005857)
- mGPS: Microbiome Global Population Structure [REF]
- Ensemble learning and model references [REF]

## Acknowledgements

This work was supervised by Eran Elhaik, with valuable input from Bijan Mousavi and Sreejith.

---

For questions or contributions, please open an issue or contact the