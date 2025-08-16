# mGPS Optimization

## BINP37: Research Project (15 cr)

This repository contains code, data, and documentation for hierarchical machine learning approaches to predict the geographic origin of environmental metagenomic samples.  
The project builds on the MetaSUB dataset and advances the state-of-the-art in microbiome geolocation, surpassing previous tools such as mGPS.

---

## Author

- [@ChandrashekarCR](https://github.com/ChandrashekarCR)

---

## Overview

Accurate prediction of sample origin from microbial signatures has important applications in biosurveillance, forensic science, and public health.  
Previous methods, such as mGPS, relied on hierarchical XGBoost models and achieved high city-level accuracy but limited coordinate precision.  
This project introduces a robust ensemble learning framework that leverages neural networks, GrowNet, gradient boosting, and transformer-based models to deliver substantial improvements in both classification and coordinate prediction.

---

## 🚀 Key Features

- **Hierarchical Prediction:**  
  Models predict continent, city, and precise coordinates in a sequential framework.
- **Ensemble Learning:**  
  Combines XGBoost, LightGBM, CatBoost, TabPFN, neural networks, and GrowNet using meta-models for optimal performance.
- **Error Propagation Analysis:**  
  Introduces a mathematical framework to quantify how misclassifications at higher levels affect coordinate predictions.
- **Advanced Feature Selection:**  
  Utilizes recursive feature elimination (RFE) and SMOTE for class imbalance correction.
- **Scalable Architecture:**  
  Designed to handle larger and more diverse datasets than previous approaches.

---

## 📊 Results

- **Tenfold Reduction in Median Coordinate Error:**  
  Ensemble model achieves a median error of **13.7 km**, compared to **137 km** for mGPS.
- **High Classification Accuracy:**  
  95% continent accuracy and 93% city accuracy on the MetaSUB dataset.
- **Robustness to Class Imbalance:**  
  Ensemble approach maintains strong performance even for underrepresented regions.
- **Fine-Scale Localization:**  
  Capable of distinguishing sample origins at the level of neighborhoods or districts within cities.

---

## 📁 Repository Structure

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

---

## 🏁 Getting Started

**1. Clone the repository:**
```sh
git clone https://github.com/ChandrashekarCR/mgps_optimization.git
cd mgps_optimization
```

**2. Set up the environment:**
```sh
conda env create -f environment.yml
conda activate binp37_env
```

**3. Data Preparation:**
- Download and preprocess the MetaSUB dataset as described in  
  `/scripts/data_preprocess/preprocess_metasub.py`.

<table>
  <thead>
    <tr>
      <th>Index</th><th>Species A</th><th>Species B</th><th>Species C</th><th>Species D</th>
      <th>Continent</th><th>City</th><th>Latitude</th><th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td><td>0.12</td><td>0.34</td><td>0.56</td><td>0.78</td>
      <td>Europe</td><td>Paris</td><td>48.8566</td><td>2.3522</td>
    </tr>
    <tr>
      <td>2</td><td>0.23</td><td>0.45</td><td>0.67</td><td>0.89</td>
      <td>Asia</td><td>Tokyo</td><td>35.6895</td><td>139.6917</td>
    </tr>
    <tr>
      <td>3</td><td>0.31</td><td>0.21</td><td>0.41</td><td>0.61</td>
      <td>North Am.</td><td>New York</td><td>40.7128</td><td>-74.0060</td>
    </tr>
    <tr>
      <td>4</td><td>0.15</td><td>0.25</td><td>0.35</td><td>0.55</td>
      <td>Africa</td><td>Nairobi</td><td>-1.2921</td><td>36.8219</td>
    </tr>
  </tbody>
</table>

> **Note:**  
> The values in this table are illustrative and provided solely to help readers understand the data format.  
> They do not represent the true values for any samples in the dataset.

- Filter and select features using RFE.

**4. Model Training:**
- Train separate neural networks, combined neural networks, GrowNet, and ensemble models using scripts and notebooks provided.
- Hyperparameter optimization is performed using Optuna.

**5. Evaluation:**
- Evaluate models using classification accuracy, F1-score, geodesic error, in-radius accuracy, AUC, and AUPR.
- Compare results with previous state-of-the-art (mGPS).

---

## 📚 References

- [MetaSUB Project](https://www.sciencedirect.com/science/article/pii/S0092867421005857)
- mGPS: Microbiome Global Population Structure [REF]
- Ensemble learning and model references [REF]

---

## 🙏 Acknowledgements

> This work was supervised by Eran Elhaik, with valuable input from Bijan Mousavi and Sreejith.

> For questions or contributions, please open an issue or contact the author.

---