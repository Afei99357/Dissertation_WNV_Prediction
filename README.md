⸻

West Nile Virus Human Case Prediction (Dissertation Project)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/machine%20learning-WNV%20Prediction-orange.svg" alt="Project">
  <img src="https://img.shields.io/badge/made%20with-%E2%9D%A4-red.svg" alt="Made with Love">
</p>

This repository contains machine learning models and analysis code developed to predict human West Nile Virus (WNV) cases based on environmental, climatic, and land-use features across the United States and California.

The project is organized into national-level models (CDC data) and regional models (California-specific data), with additional supplemental materials included.

⸻

Repository Structure
```
.
├── README.md # (this file)
├── national_CDC_code # National CDC data-based modeling
│ ├── data_demo # Example national CDC datasets
│ ├── get_annual_climate_and_landuse_cdc.py # Code to get climate/land use data for CDC data
│ └── models # ML models at the national level
│ ├── HGBR # Histogram-based gradient boosting regression
│ ├── RF # Random Forest
│ ├── SVM # Support Vector Machine
│ ├── downsample # Subsampling experiments (addressing class imbalance)
│ ├── neural_network # Neural network model
│ └── state_models # State-specific modeling
├── regional_California_code # California-specific modeling
│ ├── data_demo # California human WNV datasets
│ └── models
│ ├── iterative_models # Iterative (auto-regressive) models: HGBR, RF, SVM
│ ├── lagged_feature_enriched_model # Lagged feature-based models
│ └── spatial_model # Spatial WNV modeling
└── supplemental_materials # Supplemental figures and tables
```
⸻

Project Overview

```
National Models (national_CDC_code)
• Predicts annual human WNV cases across the U.S. using CDC-reported data.
• Machine learning methods include:
• Random Forest (RF)
• Support Vector Machine (SVM)
• Histogram-Based Gradient Boosting (HGBR)
• Neural Networks
• Subsampling experiments were conducted to address class imbalance.
• State-level models were also built to capture regional differences.

California Regional Models (regional_California_code)
• Focuses on 13 counties in California with detailed monthly WNV case reports.
• Methods include:
• Lagged feature-enriched models: use differencing and lag features to improve predictions.
• Iterative models: predict each month iteratively based on previous months.
• Spatial models: integrate county-level spatial patterns into modeling.
• Models used: SVM, RF, HGBR.
• Feature importance is analyzed using SHAP (SHapley Additive exPlanations).

Supplemental Materials (supplemental_materials)
• Contains figures and tables that support the analyses presented in the dissertation.
```
⸻
```
Data Information
• The national CDC datasets in data_demo/ are randomly generated mock data for demonstration and code testing purposes — not real human WNV case data.
• The California regional datasets (CA_human_data_2004_to_2023_final_all_counties_CDPH_scraped.csv) were scraped from California Department of Public Health West Nile Virus Reports.
• Please note:
```
**The real data used in the dissertation differs from the demo data provided in this repository.**

⸻
```
Main Features
• Automated hyperparameter optimization using Hyperopt.
• Bootstrapping to assess prediction confidence intervals (R² and MSE).
• SHAP-based feature importance at both global and local levels.
• Structured pipelines for training, validation, testing, and evaluation.
```
⸻

Requirements

Install the required Python packages by running:

pip install -r requirements.txt
```
Major dependencies include:
• pandas
• numpy
• matplotlib
• scikit-learn
• shap
• hyperopt
• torch
• plotly
• seaborn
• xarray
• opencv-python (only required if using OpenCV-based extensions; not essential for core WNV modeling)

Other:
• pathlib (built-in for Python 3.6+)
• stack_data (optional, used in debugging tools)
```
⸻

How to Run

Each script can be run independently.

For example, to run the lagged feature enriched model for California:

python regional_California_code/models/lagged_feature_enriched_model/lagged_feature_enriched_model.py

Outputs (bootstrapping results, shap plots, metrics, etc.) will be saved automatically under the corresponding results/ folders.

⸻

License

This repository is licensed under the MIT License.

MIT License

Copyright (c) 2025 Yunfei Liao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

⸻

Citation

If you use or adapt this code for your research, please cite:

Yunfei Liao. HARNESSING COMPUTATIONAL TOOLS AND COMPLEX BIOLOGICAL DATA FOR ENVIRONMENTAL AND HEALTH APPLICATIONS.
Ph.D. Dissertation. University of North Carolina at Charlotte, 2025.
