# sparkle-movie

Projet PySpark pour analyser le dataset MovieLens et construire un moteur de recommandation.

## 

Structure initiale mise en place:

- `src/` pour le code Python
- `data/raw/` pour les fichiers MovieLens bruts
- `data/processed/` pour les sorties de préparation
- `reports/figures/` pour les graphiques
- `notebooks/` pour l'exploration interactive
- `.env` et `.env.example` pour la configuration locale

## Installation

1. Crée un environnement Python dédié.
2. Installe les dépendances avec `pip install -r requirements.txt` ou `pip install -e .`.
3. Vérifie que `.env` contient bien le chemin du dataset et les paramètres Spark.
4. Vérifie aussi que `JAVA_HOME` pointe vers un JDK 17 compatible.

## Lancement de test

```bash
python -m sparkle_movie
```

Cette commande démarre une session Spark locale et confirme que la configuration de base fonctionne.

## Preparation et chargement MovieLens

Pour telecharger automatiquement le dataset officiel GroupLens (selon la taille definie dans .env), puis charger ratings.csv et movies.csv dans Spark avec un apercu des 10 premieres lignes:

```bash
python -m sparkle_movie --prepare-data
```

Comportement attendu:

- si les fichiers existent deja dans le dossier configure, ils sont reutilises
- sinon le zip est telecharge puis decompresse dans data/raw
- un nettoyage minimal est applique (valeurs manquantes, doublons, notes hors plage)

## Etape 4 - Analyse complete et livrables

Pour executer une analyse complete et reproductible (exploration, nettoyage detaille, tendances, visualisations, exports Tableau):

```bash
python -m sparkle_movie --run-analysis
```

Livrables generes:

- Rapports markdown dans `reports/`
- Figures dans `reports/figures/`
- Exports Tableau dans `data/processed/tableau/`
- Donnees nettoyees en parquet dans `data/processed/cleaned/`

##  Modelisation et comparaison des recommandations

Pour executer:

- ALS (Spark MLlib) avec ajustement d'hyperparametres
- recommandation basee contenu (TF-IDF + cosinus sur les genres)
- recommandation basee proximite utilisateurs (UserKNN)
- recommandation hybride ponderee (fusion ALS + contenu + UserKNN)
- evaluation comparative (RMSE, precision@K, recall@K, couverture)
- recommandations pour 5 utilisateurs fictifs

```bash
python -m sparkle_movie --run-modeling
```

Sorties principales:

- `reports/modeling_report.md`
- `data/processed/modeling/metrics_comparison.csv`
- `data/processed/modeling/recommendations_all_methods.csv`
- `data/processed/modeling/recommendations_fictive_users.csv`
- `data/processed/modeling/hybrid_best_weights.csv`
- `data/processed/modeling/hybrid_weight_grid_search.csv`
- `reports/figures/recommender_metrics_comparison.png`

## Synthese finale

Le projet a abouti a un pipeline complet et reproductible pour MovieLens:

- chargement et nettoyage des donnees MovieLens officiels
- analyse exploratoire avec statistiques completes, medians, quartiles et visualisations
- modelisation comparee avec ALS, contenu, UserKNN et hybridation
- optimisation des poids hybrides par grille de recherche

Resultat principal:

- le meilleur hybride retenu est `ALS=0.1`, `Content=0.1`, `UserKNN=0.8`
- Precision@10: `0.0149`
- Recall@10: `0.0217`
- Coverage@10: `0.1532`

Interpretation rapide:

- ALS reste le plus utile pour la prediction de notes et la composante collaborative pure
- UserKNN donne ici les meilleurs resultats en precision et rappel top-k
- le modele hybride optimise atteint le niveau de UserKNN tout en conservant une fusion multi-signaux plus robuste
