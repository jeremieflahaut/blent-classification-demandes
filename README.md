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

- représentation TF-IDF des textes ;
- réseau de neurones dense avec `C` sorties et activation softmax.

### 3. Validation

- Weighted F1 Score sur un jeu de test non vu à l'entraînement ;
- matrice de confusion sur l'ensemble des classes.

## Résultats

| Indicateur | Valeur |
|---|---|
| Observations après nettoyage | 260 356 (sur 300 000) |
| Doublons supprimés | 39 644 (13 %) |
| Nombre de catégories `C` | 21 |
| Part de la classe majoritaire | 39 % |
| **Weighted F1 Score** | _à compléter_ |

Le déséquilibre entre classes est important : de 101 265 observations pour la catégorie
la plus fréquente à 2 pour la plus rare. C'est ce qui justifie une métrique pondérée,
et c'est pourquoi la matrice de confusion est fournie séparément du score — elle seule
montre qu'aucune catégorie n'a été abandonnée.

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

_À compléter : traitement des libellés redondants, architecture retenue, gestion du
déséquilibre des classes._

## Développement

```bash
uv sync                              # installe les dépendances
uv run jupyter lab                   # lancer le notebook
uv export --no-hashes -o requirements.txt   # régénérer requirements.txt
```

Le notebook est versionné **avec ses sorties** : les tableaux, graphiques et la matrice
de confusion constituent les livrables du projet et doivent rester lisibles sans
exécution.
