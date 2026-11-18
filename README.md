# Demande synthétique : données, modèles et évaluation

Projet personnel réalisé avec une assistance IA, en novembre 2026. Le code prépare des données de demande et de prix **entièrement simulées**, entraîne des modèles et compare leurs résultats sur des périodes séparées. Il n'utilise aucun client, billet, transaction ou système de production réel.

## Ce qui fonctionne

- **Pandas / NumPy** : préparation, valeurs manquantes, encodage des catégories, variables cycliques et standardisation apprise uniquement sur l'entraînement.
- **Pandera** : validation du schéma, des identifiants uniques, des valeurs finies, des prix et de la demande.
- **PyTorch** : deux perceptrons de régression, cible brute ou logarithmique, et deux classifieurs d'événement rare, avec ou sans pondération des classes.
- Comparaison avec une médiane et une régression ridge NumPy ; RMSE, MAE, exactitude, précision, rappel, F1 et matrices de confusion.
- Intervalles bootstrap appariés pour comparer les erreurs ; comparaison des performances sur une population simulée décalée.
- PSI avec classes fixées sur l'entraînement et statistique KS ; le seuil PSI est une **heuristique**, sans valeur p.
- Résidualisation croisée de type DML pour retrouver une élasticité **connue dans le simulateur**, avec facteurs de confusion observés et hypothèses explicites.
- **SQLAlchemy / SQLite** : enregistrement local des prédictions puis vérification du nombre de lignes.
- **Plotly** : rapport HTML interactif ; il charge Plotly.js depuis son CDN et nécessite donc une connexion pour afficher les graphiques.
- **MLflow** : option de suivi local des paramètres, mesures et rapport, avec lecture de contrôle du résultat stocké. Aucun serveur distant ni registre de modèles n'est utilisé.

## Lancer

Environnement vérifié : Python 3.12.14. Les versions directes testées sont figées dans `requirements.txt`. Ce projet ne prétend pas avoir été validé avec Python 3.14.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.venv\Scripts\python -m unittest -v
.venv\Scripts\python experiment.py --output results --mlflow
```

Sous Linux/macOS, remplacer `.venv\Scripts\python` par `.venv/bin/python`. L'option `--mlflow` peut être omise : les modèles, mesures, CSV, SQLite et graphiques sont tout de même générés. Aucun GPU, modèle téléchargé, API payante ou secret n'est nécessaire. L'installation des bibliothèques utilise Internet ; le calcul s'exécute localement.

## Données et séparation

Le jeu de référence contient 240 jours, quatre liaisons fictives et deux segments : 1 920 lignes. Les 144 premiers jours servent à l'entraînement, les 48 suivants à la validation et les 48 derniers au test. Les identifiants sont disjoints. Une seconde population est générée avec des prix, promotions et facteurs de marché modifiés ; ses cibles sont recalculées par le même mécanisme, et non conservées artificiellement.

La médiane d'imputation, la moyenne/échelle des variables, le seuil d'événement rare, les poids de classe et les classes PSI sont calculés **sur l'entraînement**. La validation sélectionne l'itération de chaque réseau. Les données de test ne choisissent ni les variables ni les hyperparamètres.

La demande est continue et positive : il ne s'agit pas d'un simulateur complet de réservations, de capacités, de stocks ou de tarification optimale. Le projet étudie une estimation conditionnelle de demande sur des périodes futures, pas une prévision multi-horizon en production.

## Fichiers

- `experiment.py` : génération, contrôles, apprentissage, évaluation, suivi et rapports.
- `test_experiment.py` : invariants des données, absence de fuite, calculs vérifiables à la main, causalité simulée et rapprochement des sorties.
- `results/report.json` : configuration, versions, empreintes des jeux, mesures, audit des plis et limites.
- `results/predictions.csv` : 768 prédictions de test et de population décalée.
- `results/dashboard.html` : graphiques Plotly.
- `results/results.sqlite` et `results/tracking.sqlite` : bases locales générées, exclues du dépôt.

Voir [RESULTATS.md](RESULTATS.md) pour les valeurs obtenues et [METHODOLOGIE.md](METHODOLOGIE.md) pour les formules, hypothèses et limites.

## Limites

Les résultats concernent un seul mécanisme synthétique et une réalisation d'apprentissage par modèle. Les intervalles par rééchantillonnage d'exemples ne couvrent ni le réentraînement ni les dépendances temporelles. Un indicateur de dérive n'implique pas automatiquement une baisse de performance. La pondération peut améliorer le rappel en dégradant la précision. La cible logarithmique n'est pas supposée toujours supérieure.

L'estimation causale utilise une équation connue, des facteurs observés et des données manquantes au hasard. Elle ne démontre aucun effet de prix réel. Aucun déploiement PostgreSQL, Kubernetes, Argo, GitLab CI, service de prédiction ou registre de modèles n'a été exécuté. Les comptes, données et infrastructure d'un employeur ne sont pas impliqués.

## Références primaires

- [Pandera : schémas de données](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html).
- [Plotly : export HTML interactif](https://plotly.com/python/interactive-html-export/).
- [SQLAlchemy : moteurs et connexions](https://docs.sqlalchemy.org/en/21/core/engines.html).
- [MLflow : suivi d'expériences](https://mlflow.org/docs/latest/ml/tracking/).
- Chernozhukov et al., [Double/Debiased Machine Learning for Treatment and Causal Parameters](https://arxiv.org/abs/1608.00060).
