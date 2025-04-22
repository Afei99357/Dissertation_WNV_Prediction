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

.
├── README.md
├── national_CDC_code
│ ├── data_demo
│ ├── get_annual_climate_and_landuse_cdc.py
│ └── models
├── regional_California_code
│ ├── data_demo
│ └── models
└── supplemental_materials

⸻

Project Overview

National Models (national_CDC_code):
• Predicts annual human WNV cases across the U.S. using CDC-reported data.
• Machine learning methods include:
• Random Forest (RF)
• Support Vector Machine (SVM)
• Histogram-Based Gradient Boosting Regression (HGBR)
• Neural Networks
• Subsampling experiments address class imbalance.
• State-level models also capture regional variation.

California Regional Models (regional_California_code):
• Focuses on 13 counties in California with detailed monthly WNV case reports.
• Methods:
• Lagged feature enriched models (using differencing and lag features)
• Iterative prediction frameworks (auto-regressive models)
• Spatial models incorporating county-level patterns
• Models used: SVM, RF, HGBR.
• Feature importance analyzed using SHAP (SHapley Additive exPlanations).

Supplemental Materials (supplemental_materials):
• Supporting figures and tables referenced in the dissertation.

⸻

Main Features
• Automated hyperparameter optimization using hyperopt.
• Bootstrapping for confidence interval estimation (R² and MSE).
• SHAP-based feature importance analysis (global and local).
• Structured pipeline for training, validation, testing, and evaluation.

⸻

Requirements

Install required Python packages with:

pip install pandas numpy matplotlib scikit-learn shap hyperopt torch

Additional notes:
• pathlib is standard in Python 3.6+.
• stack_data is imported but not actively used — safe to ignore.

⸻

How to Run

Example: Run the lagged feature enriched model for California:

python regional_California_code/models/lagged_feature_enriched_model/lagged_feature_enriched_model.py

All outputs (bootstrapping results, SHAP plots, evaluation metrics) will be saved automatically under corresponding results/ directories.

⸻

About the Data
• National Data (national_CDC_code/data_demo):
• The datasets provided are randomly generated mock data for demonstration purposes only.
• They do not represent real CDC-reported WNV cases.
• California Data (regional_California_code/data_demo):
• The datasets were scraped from the California Department of Public Health (CDPH) official website: westnile.ca.gov/resources_reports.
• Important:
• The data here may differ slightly from the datasets used in the official dissertation results.
• Additional preprocessing and corrections were performed during the dissertation work but are not reflected in this repository.

⸻

Citation

If you use or adapt this code, please cite:

Yunfei Liao.
HARNESSING COMPUTATIONAL TOOLS AND COMPLEX BIOLOGICAL DATA FOR ENVIRONMENTAL AND HEALTH APPLICATIONS.
Ph.D. Dissertation. University of North Carolina at Charlotte, 2025.

⸻

License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2025 Yunfei Liao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
