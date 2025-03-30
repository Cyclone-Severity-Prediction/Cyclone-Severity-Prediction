# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os


# Fonction pour charger ou entraîner le modèle
@st.cache_resource
def load_or_train_model():
    # Vérifier si le modèle est déjà sauvegardé
    model_path = "random_forest_model.pkl"
    scaler_path = "scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    # Charger les données
    df = pd.read_csv(
        "/app_development/ibtracs.csv",
        low_memory=False,
        na_values=[],
        keep_default_na=False,
    ).drop(0)

    # Pré-sélection des colonnes pertinentes
    columns_to_keep = [
        "SID",
        "BASIN",
        "SUBBASIN",
        "NAME",
        "LAT",
        "LON",
        "WMO_WIND",
        "WMO_PRES",
        "DIST2LAND",
        "TD9636_STAGE",
        "NATURE",
        "STORM_SPEED",
        "STORM_DIR",
    ]
    df = df[columns_to_keep]

    # Convertir les types de données
    convert_to_num = [
        "LAT",
        "LON",
        "WMO_WIND",
        "WMO_PRES",
        "DIST2LAND",
        "STORM_SPEED",
        "STORM_DIR",
    ]
    df[convert_to_num] = df[convert_to_num].apply(pd.to_numeric, errors="coerce")
    df[["BASIN", "SUBBASIN", "NATURE", "TD9636_STAGE"]] = df[
        ["BASIN", "SUBBASIN", "NATURE", "TD9636_STAGE"]
    ].astype("str")

    # Nettoyer TD9636_STAGE et supprimer les lignes manquantes
    df["TD9636_STAGE"] = df["TD9636_STAGE"].str.strip()
    df = df[df["TD9636_STAGE"] != ""].dropna(subset=["TD9636_STAGE"])

    # Feature Engineering
    df = df.sort_values(by=["SID"])
    df["WMO_WIND_DIFF"] = df.groupby("SID")["WMO_WIND"].diff().fillna(0)
    df["WMO_PRES_DIFF"] = df.groupby("SID")["WMO_PRES"].diff().fillna(0)
    df["WMO_WIND_THRESHOLD"] = (df["WMO_WIND"] >= 64).astype(int)

    # Calcul de la distance déplacée (Haversine)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df["LAT_PREV"] = df.groupby("SID")["LAT"].shift(1)
    df["LON_PREV"] = df.groupby("SID")["LON"].shift(1)
    df["DISTANCE_MOVED"] = df.apply(
        lambda row: (
            haversine(row["LAT"], row["LON"], row["LAT_PREV"], row["LON_PREV"])
            if pd.notnull(row["LAT_PREV"])
            else 0
        ),
        axis=1,
    )
    df = df.drop(["LAT_PREV", "LON_PREV"], axis=1)

    # Supprimer les lignes avec des valeurs manquantes dans WMO_WIND et WMO_PRES
    df = df.dropna(subset=["WMO_WIND", "WMO_PRES"])

    # Features et cible
    num_features = [
        "LAT",
        "LON",
        "WMO_WIND",
        "WMO_PRES",
        "DIST2LAND",
        "STORM_SPEED",
        "STORM_DIR",
        "WMO_WIND_DIFF",
        "WMO_PRES_DIFF",
        "WMO_WIND_THRESHOLD",
        "DISTANCE_MOVED",
    ]
    cat_features = ["BASIN", "SUBBASIN", "NATURE"]
    target = "TD9636_STAGE"
    X = df.drop(columns=[target, "SID", "NAME"])
    y = df[target]

    # Séparer les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Encoder les variables catégoriques
    X_train = pd.get_dummies(X_train, columns=cat_features)
    X_test = pd.get_dummies(X_test, columns=cat_features)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # StandardScaler pour les features numériques
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    # Définir les poids des classes
    class_weights = {"0": 1, "1": 1, "2": 2, "3": 10, "4": 5, "5": 3, "6": 2}
    class_weights_dict = {cls: class_weights[cls] for cls in y.unique()}

    # Entraîner le modèle Random Forest
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        class_weight=class_weights_dict,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=0,
    )
    model.fit(X_train, y_train)

    # Sauvegarder le modèle et le scaler
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler


# Charger le modèle et le scaler
model, scaler = load_or_train_model()

# Interface Streamlit
st.title("🌪️ Tropical Cyclone Severity Prediction")
st.markdown(
    """
This application predicts the severity of a tropical cyclone based on its characteristics using a Random Forest model trained on the IBTrACS dataset.  
Enter the cyclone features below to get a prediction.
"""
)

# Dictionnaire pour mapper les prédictions
stage_mapping = {
    "0": "Tropical Disturbance",
    "1": "Tropical Depression (<34 knots)",
    "2": "Tropical Storm (34-63 knots)",
    "3": "Hurricane/Typhoon (HU1, 64 knots)",
    "4": "Hurricane/Typhoon (HU2+, >64 knots)",
    "5": "Extratropical",
    "6": "Dissipating",
}

# Formulaire pour les entrées utilisateur
st.header("Enter Cyclone Characteristics")

with st.form("cyclone_form"):
    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input(
            "Latitude (LAT)", min_value=-90.0, max_value=90.0, value=20.0
        )
        lon = st.number_input(
            "Longitude (LON)", min_value=-180.0, max_value=180.0, value=-50.0
        )
        wmo_wind = st.number_input(
            "WMO Wind Speed (knots)", min_value=0.0, max_value=200.0, value=50.0
        )
        wmo_pres = st.number_input(
            "WMO Pressure (hPa)", min_value=800.0, max_value=1100.0, value=980.0
        )
        dist2land = st.number_input(
            "Distance to Land (km)", min_value=0.0, max_value=10000.0, value=500.0
        )
        storm_speed = st.number_input(
            "Storm Speed (knots)", min_value=0.0, max_value=50.0, value=10.0
        )

    with col2:
        storm_dir = st.number_input(
            "Storm Direction (degrees)", min_value=0.0, max_value=360.0, value=270.0
        )
        wmo_wind_diff = st.number_input(
            "WMO Wind Difference (knots)", min_value=-100.0, max_value=100.0, value=0.0
        )
        wmo_pres_diff = st.number_input(
            "WMO Pressure Difference (hPa)",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
        )
        basin = st.selectbox("Basin", ["NA", "EP", "WP", "NI", "SI", "SP", "SA"])
        subbasin = st.selectbox(
            "Subbasin", ["MM", "NA", "CS", "GM", "BB", "AS", "WA", "EA"]
        )
        nature = st.selectbox("Nature", ["TS", "ET", "DS", "MX", "NR"])

    submitted = st.form_submit_button("Predict")

# Prédiction
if submitted:
    # Créer un DataFrame avec les entrées utilisateur
    input_data = pd.DataFrame(
        {
            "LAT": [lat],
            "LON": [lon],
            "WMO_WIND": [wmo_wind],
            "WMO_PRES": [wmo_pres],
            "DIST2LAND": [dist2land],
            "STORM_SPEED": [storm_speed],
            "STORM_DIR": [storm_dir],
            "WMO_WIND_DIFF": [wmo_wind_diff],
            "WMO_PRES_DIFF": [wmo_pres_diff],
            "WMO_WIND_THRESHOLD": [1 if wmo_wind >= 64 else 0],
            "DISTANCE_MOVED": [
                0.0
            ],  # Valeur par défaut (peut être calculée si données historiques disponibles)
            "BASIN": [basin],
            "SUBBASIN": [subbasin],
            "NATURE": [nature],
        }
    )

    # Encoder les variables catégoriques
    input_data = pd.get_dummies(input_data, columns=["BASIN", "SUBBASIN", "NATURE"])

    # Aligner les colonnes avec celles du modèle
    model_features = model.feature_names_in_
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    # StandardScaler pour les features numériques
    num_features = [
        "LAT",
        "LON",
        "WMO_WIND",
        "WMO_PRES",
        "DIST2LAND",
        "STORM_SPEED",
        "STORM_DIR",
        "WMO_WIND_DIFF",
        "WMO_PRES_DIFF",
        "WMO_WIND_THRESHOLD",
        "DISTANCE_MOVED",
    ]
    input_data[num_features] = scaler.transform(input_data[num_features])

    # Faire la prédiction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Trouver la probabilité associée à la catégorie prédite
    predicted_proba = (
        prediction_proba[int(prediction)] * 100
    )  # Convertir en pourcentage

    # Afficher le résultat
    st.header("Prediction Result")
    st.success(f"Predicted Cyclone Stage: **{stage_mapping[prediction]}**")
    st.info(f"Probability of this prediction: **{predicted_proba:.2f}%**")
