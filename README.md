# Classification des demandes au service client

Classification automatique de réclamations clients rédigées en langage naturel vers
une catégorie unique — réseau de neurones dense sur représentation TF-IDF.

**Stack :** Python · scikit-learn · TensorFlow (Keras)

Projet du module Deep Learning — formation Blent LLM Engineer.

## Prérequis

- Python 3.12 (TensorFlow ne publie pas de paquet pour Python 3.14)

## Installation

```bash
git clone https://github.com/jeremieflahaut/blent-classification-demandes.git
cd blent-classification-demandes
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Alternative si vous utilisez [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

## Les données

Le jeu de données (320 Mo) **n'est pas versionné**. Placez le fichier fourni dans le
dossier `data/`, sous le nom attendu par le notebook :

```
data/complaints.csv
```

Il contient 300 000 réclamations, avec deux colonnes :

| Colonne | Rôle |
|---|---|
| `Consumer complaint narrative` | le texte écrit par le client (variable explicative) |
| `Product` | la catégorie associée (variable réponse) |

Deux caractéristiques à connaître avant de lire le notebook :

- **Le texte est en anglais** et les catégories sont des produits financiers
  (`Debt collection`, `Mortgage`, `Student loan`…), et non les catégories d'assurance
  décrites dans l'énoncé. La méthode reste identique ; seuls les mots vides utilisés
  changent (anglais au lieu du français).
- **Le jeu a été anonymisé** : noms, dates et numéros de compte sont remplacés par des
  suites de `X` (`XXXX`, `XX/XX/XXXX`). Leur retrait fait partie du prétraitement.

## Lancer le notebook

Tout le travail tient dans un notebook unique :

```bash
uv run jupyter lab              # ou ouvrir classification.ipynb dans VS Code
```

Le notebook est exécutable de bout en bout (**Restart → Run All**) et suit les trois
étapes de l'énoncé.

## Objectif et contraintes

- **Classification mono-label** : chaque réclamation appartient à exactement une catégorie.
- **Toutes les classes doivent être représentées** par le modèle : aucune catégorie ne
  peut être écartée, y compris les plus rares.
- **Métrique : Weighted F1 Score > 75 %**, soit la moyenne des F1 par classe pondérée par
  la fréquence de chaque classe :

  ```
  F1 = Σ αᵢ · F1ᵢ        avec  αᵢ = nᵢ / n
  ```

- **Matrice de confusion** fournie comme preuve de résultat.

## Démarche

### 1. Analyse exploratoire et prétraitement

- décompte des observations par catégorie (`nᵢ`), avant et après nettoyage ;
- contrôle des valeurs manquantes, de la longueur des textes et des doublons ;
- retrait des masques d'anonymisation ;
- suppression des doublons exacts.

### 2. Vectorisation et modèle

- séparation train/test (80/20, stratifiée sur la catégorie) **avant** toute
  vectorisation : le vectoriseur n'apprend son vocabulaire et ses poids IDF que sur le jeu
  d'entraînement ;
- représentation TF-IDF limitée aux 10 000 termes les plus fréquents, en écartant ceux
  présents dans moins de 5 documents, avec pondération logarithmique des occurrences ;
- réseau dense `10 000 → 256 (ReLU) → C (softmax)`, optimiseur Adam, perte
  `sparse_categorical_crossentropy` ;
- arrêt anticipé sur la perte de validation, avec restauration des meilleurs poids.

### 3. Validation

- réglage effectué sur un jeu de validation prélevé sur l'entraînement (10 %), jamais sur
  le jeu de test ;
- Weighted F1 Score sur le jeu de test, non vu à l'entraînement ni au réglage ;
- matrice de confusion sur l'ensemble des classes ;
- courbes de perte entraînement / validation.

## Résultats

| Indicateur | Valeur |
|---|---|
| Observations après nettoyage | 260 356 (sur 300 000) |
| Doublons supprimés | 39 644 (13 %) |
| Catégories d'origine | 21 |
| Catégories après regroupement | 11 |
| Weighted F1 — 21 catégories d'origine | 70,6 % |
| **Weighted F1 — 11 catégories regroupées** | **85,6 %** |

Le modèle est entraîné deux fois sur des données strictement identiques : même partition,
même représentation TF-IDF, même architecture, même critère d'arrêt. Seul l'étiquetage
change. Les **15,0 points** d'écart ne sont donc imputables qu'à la nomenclature des
catégories, et non à la modélisation — voir « Regroupement des libellés redondants ».

Le déséquilibre entre classes est important : de 101 265 observations pour la catégorie
d'origine la plus fréquente à 2 pour la plus rare. C'est ce qui justifie une métrique
pondérée, et c'est pourquoi la matrice de confusion est fournie séparément du score —
elle seule montre ce que le score agrégé dissimule. Trois réserves, en l'occurrence :

- la moyenne **non pondérée** des F1 par classe (`macro avg`) vaut 0,63, contre 0,86 pour
  la moyenne pondérée. L'écart mesure précisément l'écart de traitement entre les grandes
  et les petites catégories ;
- deux catégories résiduelles, `Debt or credit management` et `Other financial service`
  (29 et 8 observations dans le jeu de test), ne sont **jamais prédites** : leur F1 est
  nul. Elles restent portées par le modèle — un neurone de sortie leur est réservé et
  elles figurent dans la matrice de confusion — mais leurs effectifs sont trop faibles
  pour permettre un apprentissage. Elles représentent 0,07 % des observations ;
- une troisième catégorie, `Virtual currency`, n'a **aucune observation dans le jeu de
  test** : ses 2 observations sont toutes tombées côté entraînement, la stratification ne
  pouvant pas répartir une observation en deux. Elle est donc apprise mais non évaluable —
  son poids αᵢ est nul, elle ne pèse pas sur le score, et sa ligne de la matrice de
  confusion est vide. Le regroupement décrit plus bas la fait disparaître, absorbée dans
  `Money transfer or virtual currency`.

## Choix techniques

- **TF-IDF plutôt qu'une couche d'embeddings.** Le routage d'une réclamation vers un
  service se décide sur le vocabulaire employé (*mortgage*, *collection*, *credit
  report*), pas sur l'ordre des mots. Une représentation sac-de-mots est donc suffisante,
  et elle est robuste aux phrases abîmées par l'anonymisation.
- **Suppression des doublons exacts.** 13 % des lignes étaient des textes strictement
  identiques, issus de campagnes de contestation déposées en masse. Les conserver
  exposait à deux problèmes : un même texte présent dans le jeu d'entraînement **et** de
  test (fuite de données, score surévalué), et un poids démultiplié de ces textes dans la
  fonction de perte.
- **Remplacement des masques par un espace**, et non par une chaîne vide : le masque est
  parfois collé au mot suivant (`account # XXXXhas violated`), ce qui produirait des mots
  inexistants dans le vocabulaire.

### Regroupement des libellés redondants (21 → 11 catégories)

Les 21 catégories d'origine contiennent plusieurs libellés pour un même sujet : la
nomenclature a changé au cours de la collecte et les anciens intitulés coexistent avec les
nouveaux. Les trois variantes de `Credit reporting` pèsent à elles seules 55 % des
observations ; on retrouve le même phénomène sur les cartes, les prêts à la consommation
et les transferts d'argent. Aucun modèle ne peut les distinguer : les textes sont de même
nature et seule la date de dépôt les sépare — une information absente des données.

Le regroupement n'a pas été décidé a priori. Un premier modèle a été entraîné sur les 21
catégories, et sa matrice de confusion établit la redondance : 80 % des demandes
étiquetées `Credit reporting` sont prédites comme `Credit reporting, credit repair
services, or other personal consumer reports`, et 50 % pour la seconde variante. Dans
chaque groupe, le libellé le plus fréquent conserve un rappel correct tandis que ses
variantes s'effondrent. Le cas ambigu de `Consumer Loan`, ancien intitulé fourre-tout, a
été tranché de la même manière : son plus fort transfert va vers `Vehicle loan or lease`
(33 %).

Ce regroupement porte la classe majoritaire de 39 % à 55 % des observations. Le seuil de
75 % en devient mécaniquement plus accessible, puisqu'un modèle constant qui prédirait
toujours la catégorie majoritaire passerait de 39 % à 55 % de justesse. Le regroupement se
justifie par la matrice de confusion, non par le gain de score qu'il procure.

### Architecture volontairement simple

Une seule couche cachée. La perte de validation touche son minimum dès la deuxième époque
puis remonte, alors que celle d'entraînement continue de décroître : le surapprentissage
s'installe donc immédiatement, ce qui indique une capacité déjà surdimensionnée pour la
tâche — 2,56 millions de paramètres, dont 99,8 % sur la première couche du seul fait des
10 000 entrées. Il a donc été traité par arrêt anticipé plutôt que par régularisation ou
par élargissement du réseau, la cible étant atteinte.

### Déséquilibre des classes non corrigé

Ni pondération des classes ni rééchantillonnage. La métrique retenue par l'énoncé pondère
les F1 par la fréquence des classes : rééquilibrer l'entraînement améliorerait le sort des
catégories rares au détriment des plus fréquentes, donc de la métrique d'évaluation
elle-même. Le déséquilibre est assumé et documenté par la matrice de confusion plutôt que
corrigé.

### Limite résiduelle

Après regroupement, `Credit reporting` attire encore 12 à 38 % des prédictions de presque
toutes les autres catégories, et `Debt collection` jusqu'à 62 %. Il ne s'agit plus de
redondance de libellés mais d'un chevauchement thématique réel : une réclamation portant
sur un prêt impayé évoque nécessairement le recouvrement et le fichage. Ce chevauchement
n'est pas réductible par le texte seul et constitue la limite du score obtenu.

## Développement

```bash
uv sync                              # installe les dépendances
uv run jupyter lab                   # lancer le notebook
uv export --no-hashes -o requirements.txt   # régénérer requirements.txt
```

Le notebook est versionné **avec ses sorties** : les tableaux, graphiques et la matrice
de confusion constituent les livrables du projet et doivent rester lisibles sans
exécution.
