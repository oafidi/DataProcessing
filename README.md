# Delivery Time Prediction – Prédire intelligemment le temps de livraison

Un projet d'intelligence artificielle appliquée au monde réel, développé avec [Khalid Abouelfaraj], visant à prédire avec précision le **temps de livraison** à partir de données historiques, tout en exploitant un pipeline de machine learning complet, robuste et déployé.

---

## 🚀 Objectifs du projet

Dans les domaines du **e-commerce**, de la **logistique**, de la **restauration** ou des **livraisons express**, prédire correctement le temps de livraison permet de :

- ✅ Améliorer la **satisfaction client** grâce à des délais plus précis
- ✅ Optimiser la **gestion des ressources humaines et logistiques**
- ✅ Réduire les **coûts** liés aux retards ou à la mauvaise planification
- ✅ Fournir des **prévisions exploitables** dans les dashboards décisionnels

---

## Pipeline complet du projet

### 1. Collecte des données
- Plus de **40 000 observations réelles**
- Données historiques liées aux livraisons (localisation, météo, trafic, etc.)

### 2. Nettoyage & Prétraitement
- Correction d’erreurs **syntaxiques et sémantiques**
- Détection et traitement des **valeurs aberrantes** (outliers)
- Imputation **intelligente des valeurs manquantes** via :
  - `KNNImputer`
  - Autres techniques sans suppression automatique

### 3. Feature Engineering & Sélection
- Création de **nouvelles variables explicatives**
- Sélection des features les plus significatives avec des **tests statistiques** :
  - Corrélation pour les variables **quantitatives**
  - Test **ANOVA** pour les variables **qualitatives**

### 4. Préparation des données
- **Z-score scaling** des variables numériques
- **One-hot encoding** des variables catégorielles
- Réduction de dimensionnalité avec **ACP (PCA)** + analyse exploratoire :
  - Étude de la **distribution**, **symétrie**, etc.

### 5. 🤖 Entraînement des modèles
- Modèles testés :
  - `LinearRegression`
  - `KNeighborsRegressor`
  - `SVR`
  - `RandomForestRegressor`
  - `XGBoostRegressor`
- Le modèle **XGBoost** s’est révélé **le plus performant** sur notre jeu de test

### 6. Déploiement avec Streamlit
- Création d’une **interface web interactive**
- Permet aux utilisateurs (livreurs, managers, plateformes...) de **prédire en temps réel** le temps de livraison à partir de nouvelles entrées

---

## Résultats

- Meilleur modèle : `XGBoostRegressor`
- Évaluation avec le **coefficient de détermination R²**
- Excellente **généralisation sur les données test**

---

## Technologies utilisées

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn
- XGBoost
- Streamlit
- Jupyter Notebooks

---

## Ce que ce projet démontre

- Ma capacité à construire un pipeline IA **complet de A à Z**
- Mon savoir-faire en **analyse de données**, **modélisation** et **déploiement**
- Ma volonté de **résoudre des problèmes métiers concrets à fort impact**
- Ma **collaboration efficace en équipe** avec [Khalid Abouelfaraj]

---

## Opportunités & Collaboration

💬 **Vous travaillez dans la logistique, la livraison ou le transport ?**
Je suis passionné par l’**innovation utile** et cherche à mettre mes compétences au service de projets à impact.  
➡️ N’hésitez pas à me contacter pour discuter de potentielles **collaborations ou applications concrètes** de ce système !

---

## Interface – Aperçu

*![Screenshot from 2025-05-16 17-02-56](https://github.com/user-attachments/assets/bd6ba5b6-b859-4887-b8b3-3f87718710a6)
*

---


