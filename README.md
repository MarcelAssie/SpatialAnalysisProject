# Analyse Spatio-Statistique des Mobilités en Île-de-France (2017-2022)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/Status-Achevé-success?style=flat-square)
![Domain](https://img.shields.io/badge/Ecole-Geodata_Paris-orange?style=flat-square)
![Domain](https://img.shields.io/badge/Spécialisation-Geo_Data_Science-violet?style=flat-square)


Ce projet propose une analyse approfondie des flux de déplacements domicile-travail au sein de la région Île-de-France sur une période de six ans. L'étude combine des approches de science de données et d'analyse spatiale pour quantifier les déterminants du choix modal et évaluer la dynamique de compétition entre les infrastructures de transport.

## Objectifs de l'étude

Cette analyse vise à répondre à quatre questions fondamentales :


*   **Analyser les tendances temporelles** : Étudier l'évolution des parts modales entre 2017 et 2022 pour identifier les dynamiques de changement dans les comportements de mobilité.
 
*   **Quantifier la compétition TC-Voiture** : Mesurer précisément le taux de conversion entre les transports en commun et l'automobile pour valider l'hypothèse de report modal.

*   **Tester l'impact de la congestion** : Simuler une dégradation progressive des temps de parcours pour évaluer si la congestion constitue un levier significatif du changement modal.


## Données de l'étude

### Sources de données
Le projet repose sur l'exploitation de deux sources de données majeures :
*   **INSEE (Fichiers Détail MOBPRO)** : Recensements annuels des déplacements domicile-travail de 2017 à 2022.
*   **IGN (ADMIN-EXPRESS)** : Géométries officielles des communes françaises pour les calculs de distances et les représentations cartographiques.

Conformément aux bonnes pratiques de versionnage, les fichiers suivants n'ont pas été intégrés au dépôt Git en raison de leur taille importante :
*   Les couches géographiques brutes de l'IGN (`COMMUNE.*`).
*   Le fichier de travail GeoPackage (`COMMUNE_COMMUNS_DCLT.gpkg`).

Pour reproduire l'analyse, ces fichiers doivent être placés dans le répertoire `data/input/`.

## Architecture du projet

```text
├── data/
│   ├── input/                  # Données sources (MOBPRO et géométries)
│   └── output/                 # Résultats générés automatiquement
│       ├── description/        # Visualisations descriptives et cartographies
│       └── correlation_analysis/  # Graphiques de régression et statistiques
├── scripts/
│   ├── run.py                  # Point d'entrée du pipeline
│   ├── main.py                 # Orchestration du chargement et des transformations
│   ├── descriptive_analysis.py # Moteur de rendu des analyses descriptives
│   ├── utils.py                # Moteur d'analyse statistique et régressions
│   ├── structured_data.py      # Scripts de structuration des données
│   └── tests/                  # Tests expérimentaux
│       ├── __init__.py
│       ├── stats_tests.ipynb   # Tests des fonctions statistiques
│       └── test.ipynb          # Tests généraux et expérimentations
└── rapport/                    # Documentation technique et académique
```

## Méthodologie et Analyse

### Simulation de la congestion
Le projet intègre un algorithme de simulation de la durée de trajet prenant en compte :
*   Une vitesse théorique de base de 30 km/h.
*   Un facteur de congestion croissant de 1,5 % par an.
*   Une composante stochastique (bruit gaussien) simulant l'aléa du trafic réel.

### Modélisation statistique
La structure distance-mode a été modélisée via une **régression logistique**, garantissant la cohérence mathématique des parts de marché (bornées entre 0 et 1).


## Installation

### Prérequis techniques
Les bibliothèques suivantes sont nécessaires au fonctionnement du pipeline :
*   Analyse de données : `pandas`, `numpy`, `scipy`, `statsmodels`
*   Géomatique : `geopandas`, `shapely`
*   Visualisation : `matplotlib`, `seaborn`

### Exécution du pipeline
Pour lancer l'intégralité de la chaîne de traitement (chargement, calculs spatiaux, analyses statistiques et génération des visuels) :

```bash
cd scripts
python run.py
```
### Génération du rapport académique
Le rapport final est rédigé en LaTeX. Pour compiler le document et générer le PDF :

```bash
cd rapport
pdflatex rapport_complet.tex
```

## Auteurs
*   **Imane Belhafiane** - École Nationale des Sciences Géographiques (ENSG)
*   **Marcel Assie** - École Nationale des Sciences Géographiques (ENSG)

**Année universitaire :** 2025-2026