# Final Streamlit deployment package

Upload these files together to the Streamlit repository:

- `heart_disease_predictor.py`
- `final_catboost_inference.cbm`
- `inference_manifest.json`
- `requirements.txt`
- `runtime.txt`

Remove the old `cat.pkl` and `XGB.pkl` from the deployed repository after the new
application has been verified. The corrected app:

- uses the final 3,000-tree CatBoost classifier from the SMOTENC-only frozen bundle;
- reproduces the final bundle probabilities after applying the fitted aggregate
  preprocessing contract;
- applies the frozen OOF Youden threshold `0.2993037568437512`;
- requires all 12 inputs and does not silently assign default values;
- enforces the final input schema (`sleep >2 and <15`; chronic-disease count `1-13`);
- always reports the probability of the positive depressive-symptom outcome;
- displays model version, limitations, privacy wording and a non-causal SHAP explanation.

Compliance note: the inference model is generated locally from the final fitted bundle.
Do not publish it until the authors/institution have confirmed that the applicable
CHARLS data-use agreement permits release of an inference-only fitted model artifact,
or written clarification has been obtained from CHARLS.

