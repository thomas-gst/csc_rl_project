# PokeRL — Apprentissage par renforcement sur Pokémon Showdown

Agent RL entraîné à jouer à Pokémon Showdown en utilisant **poke-env**, **Gymnasium** et **Stable Baselines 3**.

---

## Prérequis

| Dépendance          | Rôle                           |
| ------------------- | ------------------------------ |
| Python 3.10+        |                                |
| `poke-env`          | Environnement Pokémon Showdown |
| `gymnasium`         | Interface standard RL          |
| `stable-baselines3` | Algorithmes DQN, PPO…          |
| `sb3-contrib`       | MaskablePPO, RecurrentPPO      |
| `hydra-core`        | Chargement de config modulaire |
| `wandb`             | Logging expérimental centralisé |
| Node.js             | Serveur local Pokémon Showdown |

```bash
pip install poke-env stable-baselines3 sb3-contrib gymnasium hydra-core wandb
```

Le serveur local doit être cloné et buildé une fois (voir `setup.md`).

---

## Structure du projet

```
csc_rl_project/
├── pokemon-showdown/      # Serveur local Showdown (cloner séparément)
├── examples/
│   └── quick_start.py     # Script de test de la librairie
└── pokerl/
  ├── config.yaml        # Fichier de transition (deprecated)
  ├── configs/           # Arborescence Hydra
  │   ├── config.yaml
  │   ├── algorithm/
  │   ├── models/
  │   ├── training/
  │   ├── reward/
  │   ├── battle/
  │   ├── server/
  │   ├── eval/
  │   └── wandb/
    ├── train.py           # Script d'entraînement
    ├── eval.py            # Script d'évaluation
    └── src/
        ├── __init__.py
        ├── environment.py # Environnement Gymnasium + Action Masking
        ├── features.py    # Extraction de l'observation (embed_battle)
        └── rewards.py     # Fonctions de récompense modulaires
```

---

## Lancement rapide

**1. Démarrer le serveur Pokémon Showdown** (depuis la racine du projet) :

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```

**2. Entraîner un agent** :

```bash
cd pokerl
python train.py
python train.py algorithm=recurrentppo
python train.py resume_path=models/best.zip
python train.py training.total_timesteps=200000
python train.py wandb.project=YOUR_PROJECT wandb.entity=YOUR_TEAM
```

**3. Évaluer un modèle entraîné** :

```bash
python eval.py eval.model_path=models/pokerl_final.zip
python eval.py eval.model_path=models/best/best_model.zip eval.opponent=heuristic eval.n_battles=200
python eval.py eval.model_path=models/pokerl_final.zip eval.render=true
```

---

## Ce qui est implémenté

### `src/features.py` — Vecteur d'observation (486 dimensions)

| Bloc                   | Contenu                                                                                       | Taille |
| ---------------------- | --------------------------------------------------------------------------------------------- | ------ |
| Pokémon actif allié    | HP + type (one-hot) + boosts de stats + statut                                                | 33     |
| Moves alliés           | 4 slots : type, puissance, précision, catégorie                                               | 88     |
| Pokémon actif adverse  | Mêmes features                                                                                | 33     |
| Moves adverses connus  | Jusqu'à 10 slots connus (les inconnues sont des np.zeros)                                     | 220    |
| Banc allié             | HP + statut pour chaque pokemon du banc (×5)                                                  | 40     |
| Banc adverse           | HP + statut pour chaque pokemon du banc (×5)                                                  | 40     |
| Entry hazards + écrans | Stealth Rock, Spikes, Toxic Spikes, Sticky Web, Reflect, Light Screen, Aurora Veil (×2 côtés) | 14     |
| Météo                  | One-hot : Soleil, Pluie, Sable, Grêle…                                                        | 8      |
| Terrain                | One-hot : Électrik, Grassy, Misty, Psychic, Trick Room…                                       | 6      |
| Gimmicks               | can_dynamax, can_mega_evolve, can_tera, can_z_move                                            | 4      |

Les boosts de stats sont normalisés entre −1 et +1 (±6 → ±1). Les PP sont normalisés entre 0 et 1 (`current_pp / max_pp`). Toutes les valeurs de HP sont normalisées entre 0 et 1. Les PP des moves adverses ne sont pas inclus car non communiqués par le serveur.

---

### `src/rewards.py` — Récompenses modulaires

Trois classes disponibles (sélection via `config.yaml`) :

| Classe             | Description                                                                       |
| ------------------ | --------------------------------------------------------------------------------- |
| `DenseReward`      | Récompense dense : variation de HP + K.O. + statuts + victoire                    |
| `AggressiveReward` | Variante offensive : plus de poids sur les K.O. adverses, moins sur les PV perdus |
| `BaseReward`       | Classe abstraite pour créer sa propre récompense                                  |

Le calcul est **différentiel** : la récompense à chaque pas est la variation de la valeur d'état courante par rapport au pas précédent, ce qui évite les récompenses éparses.

---

### `src/environment.py` — Environnement Gymnasium

- `PokeRLEnv` hérite de `SinglesEnv` (poke-env / PettingZoo)
- `embed_battle()` délègue à `features.py`
- `calc_reward()` délègue au `BaseReward` configuré
- `action_masks()` construit un masque booléen sur les 22 actions (gen8) :
  - slots 0–5 : switch vers le Pokémon i
  - slots 6–9 : move 1–4 (normal)
  - slots 10–13 : move 1–4 + Méga
  - slots 14–17 : move 1–4 + Z-move
  - slots 18–21 : move 1–4 + Dynamax
- `MaskableSingleAgentWrapper` enrobe le tout en `gymnasium.Env` mono-agent compatible SB3

---

### `train.py` — Entraînement avec choix d'algorithme

La fonction `build_model()` est une factory qui instancie l'algorithme demandé :

| Algorithme     | Librairie         | On/Off-policy              | Action Masking | Mémoire |
| -------------- | ----------------- | -------------------------- | -------------- | ------- |
| `MaskablePPO`  | sb3-contrib       | On-policy                  | ✅ natif       | —       |
| `RecurrentPPO` | sb3-contrib       | On-policy                  | ✅ natif       | LSTM    |
| `DQN`          | stable-baselines3 | Off-policy (replay buffer) | ❌             | —       |

> **Note DQN :** sans action masking, l'agent peut sélectionner des actions invalides. L'environnement retombe alors sur un move aléatoire (`strict=False`), ce qui biaise la récompense. `MaskablePPO` est conseillé pour Pokémon.

Les callbacks s'adaptent automatiquement : `MaskableEvalCallback` pour les algos masquables, `EvalCallback` standard pour DQN.

---

## Configuration — Hydra (`pokerl/configs/`)

Tous les paramètres editables sont maintenant composés via Hydra :

### Serveur

`pokerl/configs/server/localhost.yaml`

### Format de combat

`pokerl/configs/battle/gen8randombattle.yaml`

### Algorithme RL

`pokerl/configs/algorithm/*.yaml` via `algorithm=maskableppo|recurrentppo|dqn`

### Récompense

`pokerl/configs/reward/dense.yaml`

### Hyperparamètres PPO (MaskablePPO / RecurrentPPO)

`pokerl/configs/models/ppo.yaml`

### Hyperparamètres DQN

`pokerl/configs/models/dqn.yaml`

### Adversaires

`pokerl/configs/training/default.yaml` et `pokerl/configs/eval/default.yaml`

### Weights & Biases

Configurer le projet/équipe ici : `pokerl/configs/wandb/default.yaml`

```yaml
enabled: true
project: "YOUR_PROJECT"
entity: "YOUR_TEAM"
```

Ou en override CLI :

```bash
python train.py wandb.project=YOUR_PROJECT wandb.entity=YOUR_TEAM
```

| Adversaire  | Description                                           |
| ----------- | ----------------------------------------------------- |
| `random`    | Choix totalement aléatoire                            |
| `max_power` | Joue toujours le move de plus haute puissance de base |
| `heuristic` | Heuristiques simples (type advantage, statuts…)       |

---

## TensorBoard

```bash
tensorboard --logdir pokerl/logs
```

Les métriques `eval/mean_reward` et `eval/mean_ep_length` permettent de suivre la progression. Les checkpoints sont sauvegardés dans `pokerl/models/` toutes les `save_freq` steps, et le meilleur modèle dans `pokerl/models/best/`.

---

## Étendre le projet

**Nouvelle récompense** : créer une classe héritant de `BaseReward` dans `src/rewards.py`, l'ajouter à `REWARD_REGISTRY`, puis changer `reward.class` dans la config.

**Nouvelle observation** : modifier `embed_battle()` dans `src/features.py` et mettre à jour `OBSERVATION_SIZE`.

**Nouveau format** : changer `battle.format` dans la config (ex : `gen9randombattle`). Le nombre d'actions dans l'espace est recalculé automatiquement par `SinglesEnv`.
