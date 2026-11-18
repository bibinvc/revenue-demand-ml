# Protocole et interprétation

## Mécanisme générateur

Chaque ligne représente un jour, une liaison et un segment fictifs. Les facteurs observés `X` comprennent un indice de marché, une promotion, la liaison, le segment et des variables calendaires. Le prix et la demande vérifient :

```text
T = log(prix) - log(100) = g(X) + bruit_prix
Y = log(demande) = m(X) + theta*T + bruit_demande
theta = -1.3
```

Les termes de `m` et `g` sont écrits dans `make_data`. Le marché influence à la fois le prix et la demande : une régression naïve de `Y` sur `T` est donc confondue. Les bruits sont indépendants conditionnellement aux facteurs simulés. Environ 3 % des valeurs de marché sont ensuite masquées au hasard, après génération de la cible.

Cette équation est un outil pédagogique dont la vérité est accessible. Elle ne constitue pas une estimation de l'élasticité d'un marché réel, ni une politique de prix exploitable.

## Préparation et apprentissage

Les jours sont séparés dans l'ordre : 60 % entraînement, 20 % validation, 20 % test. Un identifiant n'apparaît que dans un jeu. Le préparateur apprend la médiane du marché, les catégories, moyennes et écarts-types sur l'entraînement. Les variables comprennent le logarithme du prix, le carré de l'indice de marché, un sinus hebdomadaire, un cosinus mensuel et des indicatrices catégorielles.

Les deux régressions PyTorch utilisent une architecture identique et la même graine : couches 32 puis 16 unités avec ReLU. Elles diffèrent par la cible brute ou logarithmique. La cible est centrée et réduite avec les statistiques d'entraînement ; l'inversion de la transformation est faite avant calcul de la RMSE et de la MAE. Exponentier une moyenne logarithmique ne garantit pas une moyenne sur l'échelle originale : l'évaluation sur l'échelle originale rend ce choix observable.

La référence ridge est ajustée sur les mêmes variables ; l'interception n'est pas pénalisée. La référence médiane utilise la demande médiane d'entraînement. AdamW entraîne les réseaux ; l'itération donnant la perte de validation minimale est conservée. Aucun résultat de test ne sélectionne l'itération.

Pour l'événement rare, `demande > quantile_90(entraînement)` définit l'étiquette. Le poids positif est `nombre_négatifs / nombre_positifs` d'entraînement. Les classifieurs pondéré et non pondéré utilisent un seuil de score sigmoïde fixé à 0,5. Ces scores ne sont pas calibrés ; la pondération modifie l'objectif de coût et ne produit pas automatiquement une probabilité interprétable. La comparaison doit porter sur précision, rappel et F1, pas seulement sur l'exactitude.

## Mesures

```text
RMSE = sqrt(moyenne((prediction - cible)^2))
MAE = moyenne(abs(prediction - cible))
precision = TP / (TP + FP)
rappel = TP / (TP + FN)
F1 = 2 * precision * rappel / (precision + rappel)
```

Les divisions sans dénominateur sont définies à zéro pour précision/rappel/F1 et couvertes par les tests. Les matrices de confusion sont enregistrées pour reconstruire les mesures. Les différences de RMSE entre modèles sont rééchantillonnées avec **les mêmes indices** de test : il s'agit d'un bootstrap apparié conditionnel aux modèles entraînés. Il ne tient pas compte du réentraînement, d'une recherche d'hyperparamètres ou de corrélations temporelles réelles. Les intervalles sont exploratoires, pas une garantie en production.

## Population décalée

La génération décalée augmente la moyenne du marché, la fréquence des promotions et le niveau du logarithme du prix. La demande est générée à nouveau selon l'équation. Les modèles et le préparateur restent figés. Les 48 derniers jours de cette population constituent `shifted_test`.

Les classes PSI sont les quantiles du logarithme du prix **d'entraînement**, avec extrémités infinies. Les petites fréquences sont lissées puis renormalisées. `PSI > 0.2` est un seuil choisi à l'avance, explicitement heuristique. La statistique KS est le maximum d'écart entre fonctions de répartition empiriques ; aucune valeur p n'est calculée. Un déplacement détecté peut coexister avec une meilleure RMSE si la nouvelle population est plus facile à prédire : les deux résultats sont présentés séparément.

## Élasticité par résidualisation croisée

La partie causale utilise uniquement les **cas complets d'entraînement**. Cinq plis aléatoires déterministes sont formés. Pour chaque pli tenu à l'écart, deux régressions auxiliaires linéaires sont ajustées sur les autres plis : `E[T|X]` et `E[Y|X]`. Les résidus hors pli donnent :

```text
theta_estime = moyenne(residu_T * residu_Y) / moyenne(residu_T^2)
influence_i = residu_T_i * (residu_Y_i - theta_estime*residu_T_i) / moyenne(residu_T^2)
erreur_standard = ecart_type(influence) / sqrt(n)
IC approximatif = theta_estime +/- 1.96 * erreur_standard
```

Les fonctions auxiliaires utilisent les facteurs observés et leurs transformations déterministes, sans prix ni demande parmi les variables explicatives. L'audit JSON conserve les identifiants d'ajustement et de validation de chaque pli. Un contrôle rejette l'absence de variation résiduelle du prix.

Cette illustration de résidualisation de type DML suppose un mécanisme partiellement linéaire, des facteurs de confusion observés, une variation de traitement suffisante et des valeurs manquantes au hasard. Les fonctions auxiliaires sont correctement spécifiées par construction. Une réussite sur cette simulation ne résout pas le biais de variables omises, l'endogénéité ou la validité externe sur des données métier.

## Qualité, stockage et suivi

Pandera rejette les doublons d'identifiants, champs inattendus, valeurs infinies et prix non positifs. Les prédictions sont exportées en précision flottante suffisante pour rapprocher CSV, mesures et table SQLite. SQLAlchemy crée une base locale et son nombre de lignes est relu. MLflow enregistre localement les paramètres, six mesures et un rapport ; une lecture de contrôle vérifie le statut terminé, une mesure et l'artefact. Aucun enregistrement au registre de modèles n'est effectué.

La figure Plotly représente la demande observée/estimée et la RMSE comparative. Son fichier HTML utilise le CDN Plotly.js ; la lecture des graphiques nécessite Internet. Les CSV et JSON ne nécessitent pas de réseau.
