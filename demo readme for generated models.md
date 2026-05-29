# saved_models/

This folder is **populated automatically** by the Colab training notebook.

## How to generate these files

1. Open `UNSW_NB15_IDS_Classification.ipynb` in Google Colab
2. Run all cells (Runtime → Run all)
3. Cell 15 saves all models; Cell 18 downloads them as a ZIP
4. Extract the ZIP and copy the contents of `saved_models/` here

## Expected files after training

| File | Description |
|---|---|
| `random_forest.pkl` | Trained Random Forest classifier |
| `svm.pkl` | Trained SVM classifier |
| `mlp_model.keras` | Trained MLP deep learning model |
| `scaler.pkl` | Fitted StandardScaler |
| `feature_selector.pkl` | Fitted SelectKBest feature selector |
| `label_encoder.pkl` | Fitted LabelEncoder for attack categories |
| `metadata.json` | Class names, feature list, and evaluation results |

## metadata.json structure

```json
{
  "class_names": ["analysis", "backdoor", "dos", ...],
  "n_classes": 10,
  "n_features_selected": 30,
  "selected_feature_names": ["dur", "spkts", ...],
  "results": {
    "random_forest": {"accuracy": 0.97, "f1": 0.97},
    "svm":           {"accuracy": 0.95, "f1": 0.95},
    "mlp":           {"accuracy": 0.98, "f1": 0.98}
  }
}
```

> ⚠️ These binary model files are **not committed to Git** (see `.gitignore`).
> For deployment, either upload them manually to Streamlit Cloud or use Git LFS.
