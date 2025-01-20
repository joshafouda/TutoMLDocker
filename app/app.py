# Etape 3 : Streamlit App pour Consommer le Modèle Random Forest
# Organisation du répertoire :
# - TutoMLDocker/
#   - app/
#       - app.py (ce fichier)
#   - data/
#       - batch_input.csv
#       - batch_predictions.csv
#   - models/
#       - random_forest_model.pkl
#   - Dockerfile

# Importation des bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Déterminer le chemin absolu du modèle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/random_forest_model.pkl")

# Charger le modèle
if not os.path.exists(MODEL_PATH):
    st.error(f"Le fichier {MODEL_PATH} n'existe pas. Assurez-vous d'avoir entraîné et sauvegardé le modèle.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Fonctions pour les prédictions
def validate_single_input(input_data):
    """Valide les caractéristiques d'entrée pour une prédiction unique."""
    if not isinstance(input_data, dict):
        return False, "Les données doivent être un dictionnaire."

    expected_columns = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", 
        "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
    ]

    for col in expected_columns:
        if col not in input_data:
            return False, f"Caractéristique manquante : {col}"
        if not isinstance(input_data[col], (int, float)):
            return False, f"La caractéristique '{col}' doit être un nombre."

    return True, ""

def single_prediction(input_features):
    input_array = np.array([input_features[col] for col in input_features]).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return prediction

def batch_prediction(input_file):
    input_data = pd.read_csv(input_file)

    # Validation des colonnes
    expected_columns = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", 
        "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
    ]
    if not all(col in input_data.columns for col in expected_columns):
        return None, f"Le fichier doit contenir les colonnes suivantes : {', '.join(expected_columns)}"

    # Prédictions
    predictions = model.predict(input_data)
    input_data["prediction"] = predictions

    # Créer le répertoire ../data s'il n'existe pas
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "batch_predictions.csv")
    input_data.to_csv(output_file, index=False)
    return output_file, ""


# Streamlit UI
st.title("Application de Prédiction - Random Forest")
st.sidebar.title("Menu")
menu = st.sidebar.radio("Choisissez une option", ["Single Prediction", "Batch Prediction"])

if menu == "Single Prediction":
    st.header("Prédiction Unique")

    # Création des champs de saisie
    user_input = {}
    for col in [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", 
        "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
    ]:
        user_input[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Prédire"):
        valid, message = validate_single_input(user_input)
        if valid:
            prediction = single_prediction(user_input)
            st.success(f"Classe prédite : {prediction}")
        else:
            st.error(message)

elif menu == "Batch Prediction":
    st.header("Prédiction par Lot")

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    if st.button("Prédire") and uploaded_file is not None:
        output_file, message = batch_prediction(uploaded_file)
        if output_file:
            st.success("Prédictions réalisées avec succès.")
            st.download_button(
                label="Télécharger les résultats",
                data=open(output_file, "rb").read(),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
        else:
            st.error(message)
