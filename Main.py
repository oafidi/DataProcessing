import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Dashboard XGBoost - Test de Modèle",
    page_icon="",
    layout="wide"
)

# Titre principal
st.title(" Dashboard de Test - Modèle XGBoost de Livraison")
st.markdown("---")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_xgb.pkl")
        return model
    except FileNotFoundError:
        st.error("Modèle non trouvé. Assurez-vous que 'model_xgb.pkl' est dans le répertoire.")
        return None

model = load_model()

@st.cache_data
def get_scaling_params():
    return {
        'age': {'mean': 30.5, 'std': 8.2},
        'rating': {'mean': 4.2, 'std': 0.6},
        'multiple_deliveries': {'mean': 1.5, 'std': 1.2},
        'distance': {'mean': 12.4, 'std': 7.8}
    }

scaling_params = get_scaling_params()

def preprocess_input(age, rating, multiple_deliveries, distance):
    age_scaled = (age - scaling_params['age']['mean']) / scaling_params['age']['std']
    rating_scaled = (rating - scaling_params['rating']['mean']) / scaling_params['rating']['std']
    deliveries_scaled = (multiple_deliveries - scaling_params['multiple_deliveries']['mean']) / scaling_params['multiple_deliveries']['std']
    distance_scaled = (distance - scaling_params['distance']['mean']) / scaling_params['distance']['std']
    
    return age_scaled, rating_scaled, deliveries_scaled, distance_scaled

def create_feature_vector(age, rating, multiple_deliveries, distance, weather, traffic, vehicle, festival, city, type_order, day, month_start, month_end):
    age_scaled, rating_scaled, deliveries_scaled, distance_scaled = preprocess_input(age, rating, multiple_deliveries, distance)
    
    features = np.zeros(34)
    

    features[0] = age_scaled
    features[1] = rating_scaled
    features[2] = deliveries_scaled
    features[3] = distance_scaled
    

    weather_mapping = {"Cloudy": 4, "Fog": 5, "Sandstorms": 6, "Stormy": 7, "Sunny": 8, "Windy": 9}
    if weather in weather_mapping:
        features[weather_mapping[weather]] = 1
    

    traffic_mapping = {"High": 10, "Jam": 11, "Low": 12, "Medium": 13}
    if traffic in traffic_mapping:
        features[traffic_mapping[traffic]] = 1

    vehicle_mapping = {"bicycle": 14, "electric_scooter": 15, "motorcycle": 16, "scooter": 17}
    if vehicle in vehicle_mapping:
        features[vehicle_mapping[vehicle]] = 1
    
    features[18 if festival == "No" else 19] = 1

    city_mapping = {"Metropolittan": 20, "Semi-Urban": 21, "Urban": 22}
    if city in city_mapping:
        features[city_mapping[city]] = 1
 
    type_mapping = {"Buffet": 23, "Drinks": 24, "Meal": 25, "Snack": 26}
    if type_order in type_mapping:
        features[type_mapping[type_order]] = 1
    
    day_idx = 27 + int(day)
    if 27 <= day_idx <= 33:
        features[day_idx] = 1

    
    return features.reshape(1, -1)

# Interface principale
if model is not None:

    st.sidebar.header("Paramètres de Livraison")
    

    st.sidebar.subheader("Caractéristiques du Livreur")
    age = st.sidebar.slider("Âge du livreur", 18, 60, 30)
    rating = st.sidebar.slider("Note du livreur", 1.0, 5.0, 4.5, 0.1)
    multiple_deliveries = st.sidebar.slider("Nombre de livraisons multiples", 0, 10, 1)
    
    st.sidebar.subheader("Détails de la Livraison")
    distance = st.sidebar.slider("Distance (km)", 0.1, 50.0, 10.0, 0.1)
 
    st.sidebar.subheader("Conditions")
    weather_options = ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"]
    selected_weather = st.sidebar.selectbox("Conditions météorologiques", weather_options)
    
    traffic_options = ["Low", "Medium", "High", "Jam"]
    selected_traffic = st.sidebar.selectbox("Densité du trafic", traffic_options)
    

    vehicle_options = ["bicycle", "scooter", "motorcycle", "electric_scooter"]
    selected_vehicle = st.sidebar.selectbox("Type de véhicule", vehicle_options)
    

    festival = st.sidebar.selectbox("Festival", ["No", "Yes"])
    

    city_options = ["Metropolittan", "Urban", "Semi-Urban"]
    selected_city = st.sidebar.selectbox("Type de ville", city_options)
    

    type_options = ["Meal", "Snack", "Drinks", "Buffet"]
    selected_type = st.sidebar.selectbox("Type de commande", type_options)
    

    day_options = ["0", "1", "2", "3", "4", "5", "6"]
    day_names = {"0": "Lundi", "1": "Mardi", "2": "Mercredi", "3": "Jeudi", "4": "Vendredi", "5": "Samedi", "6": "Dimanche"}
    selected_day = st.sidebar.selectbox("Jour de la semaine", day_options, format_func=lambda x: f"{x} - {day_names[x]}")
    
    month_start = st.sidebar.selectbox("Début de mois", ["No", "Yes"])
    month_end = st.sidebar.selectbox("Fin de mois", ["No", "Yes"])
    

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Faire une Prédiction", type="primary", use_container_width=True)
    
    if predict_button:
        features = create_feature_vector(
            age, rating, multiple_deliveries, distance,
            selected_weather, selected_traffic, selected_vehicle,
            festival, selected_city, selected_type, selected_day,
            month_start, month_end
        )
        
        prediction = model.predict(features)[0]
        
        st.success(f" Temps de livraison prédit: **{prediction:.2f} minutes**")

        col1, col2 = st.columns(2)
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Temps de livraison (minutes)"},
                gauge = {
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "khaki"},
                        {'range': [40, 60], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 45
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            categories = ['Très rapide', 'Rapide', 'Normal', 'Lent']
            values = [15, 25, 35, 45]
            
            pred_category = 'Très rapide' if prediction <= 20 else \
                          'Rapide' if prediction <= 30 else \
                          'Normal' if prediction <= 40 else 'Lent'
            
            fig_bar = px.bar(
                x=categories, 
                y=values,
                title="Comparaison avec les temps moyens",
                color=categories,
                color_discrete_map={
                    'Très rapide': 'green',
                    'Rapide': 'lightgreen',
                    'Normal': 'orange',
                    'Lent': 'red'
                }
            )
            
            fig_bar.add_hline(
                y=prediction, 
                line_dash="dash", 
                line_color="blue",
                annotation_text=f"Prédiction: {prediction:.1f}min ({pred_category})",
                annotation_position="top left"
            )
            
            fig_bar.update_layout(
                xaxis_title="Catégorie de livraison",
                yaxis_title="Temps (minutes)",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.header("Analyse de sensibilité")
        
        tab1, tab2, tab3 = st.tabs(["Distance", "Conditions météo", "Type de véhicule"])
        
        with tab1:
            st.subheader("Impact de la distance sur le temps de livraison")
            distances = np.linspace(1, 50, 20)
            predictions_distance = []
            
            for dist in distances:
                features = create_feature_vector(
                    age, rating, multiple_deliveries, dist,
                    selected_weather, selected_traffic, selected_vehicle,
                    festival, selected_city, selected_type, selected_day,
                    month_start, month_end
                )
                pred = model.predict(features)[0]
                predictions_distance.append(pred)
            
            fig_distance = px.line(
                x=distances, 
                y=predictions_distance,
                title="Impact de la distance sur le temps de livraison",
                labels={'x': 'Distance (km)', 'y': 'Temps prédit (minutes)'},
                markers=True
            )
            fig_distance.add_vline(x=distance, line_dash="dash", line_color="red",
                               annotation_text=f"Distance actuelle: {distance} km")
            st.plotly_chart(fig_distance, use_container_width=True)
        
        with tab2:
            st.subheader("Impact des conditions météorologiques")
            weather_predictions = {}
            for weather in weather_options:
                features = create_feature_vector(
                    age, rating, multiple_deliveries, distance,
                    weather, selected_traffic, selected_vehicle,
                    festival, selected_city, selected_type, selected_day,
                    month_start, month_end
                )
                pred = model.predict(features)[0]
                weather_predictions[weather] = pred
            
            df_weather = pd.DataFrame({
                'Conditions': list(weather_predictions.keys()),
                'Temps prédit': list(weather_predictions.values())
            })
            
            fig_weather = px.bar(
                df_weather,
                x='Conditions',
                y='Temps prédit',
                title="Impact des conditions météorologiques",
                labels={'x': 'Conditions météo', 'y': 'Temps prédit (minutes)'},
                color='Temps prédit',
                color_continuous_scale='RdYlGn_r'
            )
            
            selected_idx = df_weather[df_weather['Conditions'] == selected_weather].index[0]
            fig_weather.add_annotation(
                x=selected_weather,
                y=weather_predictions[selected_weather],
                text="Sélectionné",
                showarrow=True,
                arrowhead=1
            )
            
            st.plotly_chart(fig_weather, use_container_width=True)
        
        with tab3:

            st.subheader("Impact du type de véhicule")
            vehicle_predictions = {}
            for vehicle in vehicle_options:
                features = create_feature_vector(
                    age, rating, multiple_deliveries, distance,
                    selected_weather, selected_traffic, vehicle,
                    festival, selected_city, selected_type, selected_day,
                    month_start, month_end
                )
                pred = model.predict(features)[0]
                vehicle_predictions[vehicle] = pred
            
            df_vehicle = pd.DataFrame({
                'Véhicule': list(vehicle_predictions.keys()),
                'Temps prédit': list(vehicle_predictions.values())
            })
            
            fig_vehicle = px.bar(
                df_vehicle,
                x='Véhicule',
                y='Temps prédit',
                title="Impact du type de véhicule",
                labels={'x': 'Type de véhicule', 'y': 'Temps prédit (minutes)'},
                color='Véhicule'
            )
            
            fig_vehicle.add_annotation(
                x=selected_vehicle,
                y=vehicle_predictions[selected_vehicle],
                text="Sélectionné",
                showarrow=True,
                arrowhead=1
            )
            
            st.plotly_chart(fig_vehicle, use_container_width=True)


