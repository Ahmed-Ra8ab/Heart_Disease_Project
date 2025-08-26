
# Comprehensive Machine Learning Pipeline on Heart Disease (UCI)

## Folder Structure
```
Heart_Disease_Project/
├─ data/
│  ├─ heart_disease.csv                # Place your dataset here
│  ├─ cleaned_heart.csv                # Produced by 01
│  ├─ pca_features.csv, labels.csv     # Produced by 02
│  ├─ selected_features.csv            # Produced by 03
├─ notebooks/
│  ├─ 01_data_preprocessing.ipynb
│  ├─ 02_pca_analysis.ipynb
│  ├─ 03_feature_selection.ipynb
│  ├─ 04_supervised_learning.ipynb
│  ├─ 05_unsupervised_learning.ipynb
│  ├─ 06_hyperparameter_tuning.ipynb
├─ models/
│  ├─ final_model.pkl                  # Produced by 06
│  ├─ scaler.joblib, pca_95.joblib     # Artifacts
├─ ui/
│  ├─ app.py                           # Streamlit UI
├─ deployment/
│  ├─ ngrok_setup.txt
├─ results/
│  ├─ evaluation_metrics.txt
├─ requirements.txt
├─ README.md
├─ .gitignore
```

## Quick Start
1. Create a Python 3.10+ virtual environment.
2. Install deps: `pip install -r requirements.txt`
3. Put `heart_disease.csv` in `data/` (UCI Heart dataset).
4. Run notebooks in order (01 → 06).
5. Launch the app:
   ```bash
   streamlit run ui/app.py
   ```

> **Important**: For Streamlit, the most reliable approach is to train & save a **Pipeline** that handles preprocessing from raw inputs. The current notebooks train on selected/encoded features for clarity. You can extend notebook 06 to wrap the best model inside a `ColumnTransformer`+`Pipeline` for production.

## Dataset
UCI Heart Disease: Cleveland subset commonly used. Make sure columns match or adjust preprocessing accordingly.

## License
MIT
