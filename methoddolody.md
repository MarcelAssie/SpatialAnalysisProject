

# Méthodologie proposée


## 1. Cadrage de l’étude et formulation de la problématique

### 1.1 Problématique générale

L’objectif est d’analyser dans quelle mesure les **modifications du réseau de transport** (ouvertures, fermetures, extensions de lignes) influencent le **choix du mode de transport principal** utilisé pour les déplacements domicile–travail, à une échelle territoriale donnée.

Contrairement aux approches classiques basées sur des matrices origine–destination, l’étude se concentre sur les **flux sortants par origine**, sans connaissance explicite des destinations.

### 1.2 Hypothèse centrale

Les évolutions du réseau de transport modifient :

* l’**accessibilité potentielle** des territoires,
* la **compétitivité relative des modes** (transport collectif vs voiture, etc.),
  ce qui se traduit par une **variation mesurable de la répartition modale** des actifs résidant dans une zone donnée.

---

## 2. Définition du périmètre spatial et des réseaux étudiés

### 2.1 Zone d’étude

Définition d’un territoire cohérent avec :

* la disponibilité des données,
* la structure du réseau de transport.

Exemples :

* aire urbaine ou métropolitaine,
* ensemble de communes desservies par un même réseau structurant (RER, métro, tram).

Les unités spatiales retenues sont les **zones de résidence** (communes, IRIS, quartiers).

### 2.2 Typologie des réseaux de transport

Sélection des réseaux pertinents :

* Réseaux lourds : RER, métro, grandes lignes,
* Réseaux intermédiaires : tramway,
* Réseaux de surface : bus,
* Modes individuels : voiture, deux-roues, marche.

Chaque modification du réseau est caractérisée par :

* type (ouverture, fermeture, extension),
* date de mise en service ou d’arrêt,
* mode concerné,
* localisation spatiale.

---

## 3. Données mobilisées et préparation

### 3.1 Données de flux domicile–travail

Pour chaque zone d’origine et chaque date :

* nombre d’actifs quittant la zone,
* mode de transport principal utilisé.

Important :
Les données décrivent des **flux sortants agrégés**, sans information sur la destination finale. L’analyse porte donc sur :

* la **propension modale des résidents**,
* et non sur les relations spatiales origine–destination.

### 3.2 Données sur le réseau de transport

Historique temporel des infrastructures :

* tracés géographiques des lignes,
* localisation des stations/arrêts,
* dates d’ouverture et de fermeture.

Ces données permettent de reconstruire un **réseau évolutif dans le temps**.

### 3.3 Variables de contrôle (si disponibles)

Pour limiter les biais :

* données socio-démographiques (population active, motorisation),
* données d’emploi globales (tendance macro-économique),
* variables temporelles (année, période).

---

## 4. Construction d’indicateurs spatiaux d’accessibilité

### 4.1 Justification méthodologique

En l’absence de destinations connues, l’impact du réseau est appréhendé via des **indicateurs d’accessibilité potentielle**, calculés à partir de la zone de résidence.

L’hypothèse est que :

> une amélioration de l’accessibilité locale influence le choix du mode, indépendamment de la destination exacte.

### 4.2 Indicateurs possibles

Pour chaque zone et chaque date :

* distance ou temps d’accès au réseau structurant le plus proche,
* nombre de stations accessibles dans un rayon donné,
* densité de lignes ou d’arrêts,
* indice d’accessibilité cumulée (pondéré par le type de réseau).

Ces indicateurs sont recalculés **avant et après chaque modification du réseau**.

---

## 5. Analyse exploratoire statistique et spatiale

### 5.1 Analyse descriptive

* évolution temporelle de la répartition modale par zone,
* comparaison avant/après modification du réseau,
* cartographie des parts modales dominantes.

### 5.2 Analyse spatiale

* cartographie des indicateurs d’accessibilité,
* analyse de la spatialisation des changements modaux,
* détection de clusters territoriaux (zones réactives / non réactives).

### 5.3 Analyse de corrélation

Étude des relations entre :

* variation des indicateurs d’accessibilité,
* variation des parts modales (ex. transports collectifs).

Méthodes possibles :

* corrélations simples,
* analyses comparatives entre zones impactées et zones témoins,
* approches quasi-expérimentales (avant/après).

---

## 6. Modélisation statistique

### 6.1 Variable dépendante

Choix du mode de transport principal, exprimé :

* soit en part modale,
* soit en probabilité d’utilisation d’un mode donné.

### 6.2 Variables explicatives

* indicateurs d’accessibilité au réseau,
* caractéristiques du réseau modifié,
* variables socio-démographiques et temporelles.

### 6.3 Modèles envisageables

* régression logistique ou multinomiale,
* modèles de panel (zones × temps),
* modèles à effets fixes pour capter les spécificités territoriales.

L’absence de destination est intégrée comme une **hypothèse structurelle du modèle**, qui explique un comportement résidentiel agrégé.

---

## 7. Prédiction et scénarios

### 7.1 Simulation de modifications du réseau

À partir du modèle estimé :

* simulation de l’ouverture ou fermeture d’une ligne,
* estimation de l’évolution attendue des parts modales par zone.

### 7.2 Analyse des résultats

* identification des zones les plus sensibles,
* comparaison des effets selon le type de réseau,
* discussion sur les limites liées à l’absence de destinations.

