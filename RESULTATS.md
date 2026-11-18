# Résultats mesurés — 18 novembre 2026

Exécution locale avec Python 3.12.14, PyTorch 2.15.0+cpu, Pandas 3.0.1, NumPy 2.3.5, Pandera 0.36.0, Plotly 7.1.0, SQLAlchemy 2.1.9 et MLflow-skinny 3.18.0. Les **27 tests automatisés ont réussi**. Une nouvelle exécution complète a reproduit exactement les mesures numériques, les jeux et les itérations sélectionnées dans cet environnement ; l'identifiant MLflow et l'identifiant HTML ne sont pas des sorties déterministes.

## Échantillons et préparation

- Jeu de référence : **1 920 lignes fictives**, 240 jours, quatre liaisons et deux segments.
- Entraînement : 1 152 lignes, jours 0–143 ; validation : 384 lignes, jours 144–191 ; test : 384 lignes, jours 192–239.
- Population décalée : 384 autres lignes sur les derniers jours, avec identifiants séparés.
- Quarante valeurs de marché manquent dans l'entraînement. La médiane d'imputation est apprise sur ce jeu uniquement.
- Asymétrie empirique de la demande d'entraînement : **9,15** ; les deux stratégies de cible sont donc comparées explicitement.
- Événement rare : demande supérieure au quantile 90 d'entraînement, soit environ 117,13. Prévalence d'entraînement : **10,07 %**, poids positif : **8,93**. Sur le test, 36 exemples sur 384 sont positifs.

## Régression sur l'échelle originale de la demande

| Modèle | RMSE, test | MAE, test | RMSE, population décalée |
| --- | ---: | ---: | ---: |
| Médiane d'entraînement | 40,15 | 25,23 | 26,09 |
| Ridge NumPy | 17,49 | 11,30 | 15,93 |
| Perceptron, cible logarithmique | 9,47 | 6,63 | 6,37 |
| Perceptron, cible brute | 9,24 | 6,57 | 7,84 |

La cible brute donne ici une RMSE de test légèrement plus basse que la cible logarithmique. L'ordre s'inverse sur la population décalée : il n'y a pas de supériorité universelle revendiquée pour une transformation.

La différence de RMSE « perceptron logarithmique moins ridge » est **-8,02**, avec un intervalle bootstrap apparié d'exemples à 95 % d'environ **[-12,62 ; -4,72]**. Cet intervalle conditionnel ne couvre pas le réentraînement ni une dépendance temporelle réelle.

## Classification de demande élevée

Les scores sigmoïdes sont comparés à 0,5. Ils ne sont pas calibrés ; la pondération modifie la fonction de coût.

| Classifieur | Exactitude | Précision | Rappel | F1 | TP / TN / FP / FN |
| --- | ---: | ---: | ---: | ---: | --- |
| Pondéré | 95,31 % | 68,00 % | 94,44 % | 0,791 | 34 / 332 / 16 / 2 |
| Non pondéré | 96,88 % | 81,58 % | 86,11 % | 0,838 | 31 / 341 / 7 / 5 |

La pondération augmente le rappel, mais réduit la précision et le F1 dans cette réalisation. Elle ne constitue donc pas une amélioration générale du classifieur.

## Population décalée et contrôle des distributions

Sur le logarithme du prix, les classes PSI sont fixées par l'entraînement :

| Jeu | PSI | Statistique KS | Drapeau heuristique PSI > 0,2 |
| --- | ---: | ---: | --- |
| Test de référence | 0,016 | 0,043 | Non |
| Population décalée | 1,356 | 0,486 | Oui |

Ce contrôle est **univarié** : il ne certifie pas toutes les distributions. Le décalage est bien visible alors que la RMSE logarithmique baisse. Cela illustre qu'une dérive de variables n'implique pas automatiquement une dégradation de performance. La population décalée ne contient que six événements positifs ; ses mesures de classification sont particulièrement sensibles aux petits effectifs.

## Élasticité dans le simulateur

La résidualisation croisée utilise **1 112 cas complets d'entraînement**, répartis en cinq plis :

- Valeur connue du générateur : **-1,300**.
- Coefficient naïf, sans ajustement : **-0,631**.
- Estimateur avec résidualisation croisée : **-1,294**.
- Intervalle approximatif à 95 % : **[-1,320 ; -1,268]**.

Ce résultat dépend d'une équation connue et de fonctions auxiliaires correctement spécifiées. Il ne démontre aucun effet causal de prix sur un vrai marché. Les hypothèses sont détaillées dans [METHODOLOGIE.md](METHODOLOGIE.md).

## Tests, stockage et suivi local

Les 27 tests vérifient notamment les séparations, les statistiques apprises uniquement sur l'entraînement, le rejet des données invalides, les calculs manuels de mesures, les sorties sans division par zéro, les seuils/poids, les classes de dérive figées, la séparation des plis causaux, l'absence de variation de traitement et le rapprochement CSV/SQLite/rapport.

**768 lignes de prédictions** ont été enregistrées dans SQLite puis relues via SQLAlchemy. Les quatre états PyTorch sélectionnés en mémoire correspondent aux itérations 51, 113, 33 et 54. **Aucun fichier de checkpoint de modèle n'est écrit**, et aucun registre de modèles n'est utilisé.

Le suivi MLflow local a réellement été exécuté : statut **FINISHED**, **six mesures** stockées et un instantané de rapport présent dans les artefacts. La lecture de contrôle du statut, d'une mesure et de la présence de l'artefact a réussi. Les métadonnées sont dans [results/verification.json](results/verification.json) ; les bases SQLite et le répertoire local MLflow restent exclus du dépôt.

Les résultats détaillés sont dans [results/report.json](results/report.json), les observations dans [results/predictions.csv](results/predictions.csv) et les graphiques dans [results/dashboard.html](results/dashboard.html). Les graphiques chargent Plotly.js depuis son CDN. Aucun déploiement, serveur de prédiction ou essai sur données métier n'a été effectué.
