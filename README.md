# mGPS Optimization

## BINP37: Research Project (15 cr)

This repository contains code, data, and documentation for hierarchical machine learning approaches to predict the geographic origin of environmental metagenomic samples.  
The project builds on the MetaSUB dataset and advances the state-of-the-art in microbiome geolocation, surpassing previous tools such as mGPS.

---

## 👤 Author

- [@ChandrashekarCR](https://github.com/ChandrashekarCR)

---

## 📝 Overview

Accurate prediction of sample origin from microbial signatures is crucial for biosurveillance, forensic science, and public health.  
This project introduces a robust ensemble learning framework leveraging neural networks, GrowNet, gradient boosting, and transformer-based models for improved classification and coordinate prediction.

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

- **Median Coordinate Error:**  
  Ensemble model: **13.7 km** vs. mGPS: **137 km**
- **Classification Accuracy:**  
  95% continent, 93% city (MetaSUB dataset)
- **Robustness:**  
  Strong performance for underrepresented regions.
- **Fine-Scale Localization:**  
  Distinguishes neighborhoods/districts within cities.

---

## 📁 Repository Structure

```
binp37/
├── data/                # Raw and processed data
│   ├── metasub/         # MetaSUB dataset
│   ├── geopandas/       # Geospatial mapping data
├── notebooks/           # Jupyter notebooks
├── scripts/             # Preprocessing, modeling, feature engineering
├── report/              # LaTeX manuscript source
│   ├── abstract/
│   ├── introduction/
│   ├── methods/
│   ├── results/
│   ├── discussions/
│   ├── references/
├── .gitignore           # Version control exclusions
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

This step ensures your data is clean, consistent, and ready for modeling.

- **Preprocessing the MetaSUB Dataset**

  The main script for preprocessing is  
  `/scripts/data_preprocess/preprocess_metasub.py`.

  **Purpose:**  
  - Merges taxonomic abundance data with sample metadata.
  - Performs quality control, removing cities with insufficient samples.
  - Corrects mislabelled coordinates and harmonizes metadata fields.
  - Outputs a clean, analysis-ready CSV file.

  **Required Input Files:**  
  - Taxa abundance: `/data/metasub/metasub_taxa_abundance.csv`
  - Metadata: `/data/metasub/complete_metadata.csv`

  **How to Run:**
  ```sh
  python3 scripts/data_preprocess/preprocess_metasub.py \
    -m /data/metasub/complete_metadata.csv \
    -t /data/metasub/metasub_taxa_abundance.csv \
    -o /results/metasub/processed_metasub.csv
  ```
  - `-m`: Path to metadata file.
  - `-t`: Path to taxa abundance file.
  - `-o`: (Optional) Output file path for processed data.

  The script will:
  - Filter out poorly represented cities.
  - Fix coordinate errors.
  - Save the processed dataset for downstream analysis.

  **Example Output:**  
  `/results/metasub/processed_metasub.csv`

  <table>
    <thead>
      <tr>
        <th>Index</th><th>Species A</th><th>Species B</th><th>Species C</th><th>Species D</th>
        <th>Continent</th><th>City</th><th>Latitude</th><th>Longitude</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Sample1</td><td>0.12</td><td>0.34</td><td>0.56</td><td>0.78</td>
        <td>Europe</td><td>Paris</td><td>48.8566</td><td>2.3522</td>
      </tr>
      <tr>
        <td>Sample2</td><td>0.23</td><td>0.45</td><td>0.67</td><td>0.89</td>
        <td>Asia</td><td>Tokyo</td><td>35.6895</td><td>139.6917</td>
      </tr>
      <tr>
        <td>Sample3</td><td>0.31</td><td>0.21</td><td>0.41</td><td>0.61</td>
        <td>North Am.</td><td>New York</td><td>40.7128</td><td>-74.0060</td>
      </tr>
      <tr>
        <td>Sample4</td><td>0.15</td><td>0.25</td><td>0.35</td><td>0.55</td>
        <td>Africa</td><td>Nairobi</td><td>-1.2921</td><td>36.8219</td>
      </tr>
    </tbody>
  </table>

  > **Note:**  
  > Table values are illustrative and do not represent true dataset samples.

---

**4. Feature Selection with RFE**

After preprocessing, select the most informative features using Recursive Feature Elimination (RFE).

- **Script:**  
  `/scripts/feature_engineering/rfe_feature_selection.py`

  **Purpose:**  
  - Automatically identifies and retains the most relevant microbial species/features for prediction.
  - Reduces dimensionality and improves model performance.

  **Required Input File:**  
  - Processed metadata file from previous step (e.g., `/results/metasub/processed_metasub.csv`)

  **How to Run:**
  ```sh
  python3 scripts/feature_engineering/rfe_feature_selection.py \
    -i /results/metasub/processed_metasub.csv \
    -o /results/metasub/selected_features.csv
  ```
  - `-i`: Path to processed metadata file.
  - `-o`: (Optional) Output file path for selected features.

  The script will:
  - Apply RFE to select top features.
  - Output a reduced dataset containing only the selected features.

  **Example Output:**  
  `/results/metasub/selected_features.csv`

---

## 📚 References

- [MetaSUB Project](https://www.sciencedirect.com/science/article/pii/S0092867421005857)
- mGPS: Microbiome Global Population Structure [REF]
- Ensemble learning and model references [REF]

---

## 🙏 Acknowledgements

> Supervised by Eran Elhaik, with input from Bijan Mousavi and Sreejith.  
> For questions or contributions, open an issue or contact the author.

---