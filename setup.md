# Description
**Auteur : Thomas**

Le but du notebook est de faciliter l'installation de poke-env sur les machines du BR, ainsi que la mise en ligne d'un serveur de jeu auquel on peut accéder à distance.
Je vais donc détailler les étapes que j'ai suivies. 

# Installation de poke-env 

Perso, pour tous les projets, je commence par initialiser un venv sur les machines du BR, et j'y installe les dépendances nécessaires. 
Une fois la session ssh ouverte, je trouve un endroit pour mon projet et j'y crée un venv.
```bash
mkdir csc_rl_project
cd csc_rl_project
uv init
```
Ensuite j'ajoute le module poke-env. 
```bash
uv add poke-env
```
Puis, il faut cloner le repo pokemon-showdown (dans le dossier du projet).
```bash
git clone https://github.com/smogon/pokemon-showdown.git
```

Pour pouvoir setup la librairie et lancer des serveurs de jeu, il faut commencer par installer la bonne version de node. 
Les ordis du BR sont généralement équipés de node 16, mais poke-env nécessite la version 18.
```bash
nvm install 20
nvm use 20
```
On peut finir de setup pokemon-showdown en installant les dépendances et en build le projet.
```bash
cd pokemon-showdown
npm install
npm run build
cp config/config-example.js config/config.js
```
# Lancement d'un serveur de jeu et redirection de port
Maintenant que tout est installé, on peut lancer un serveur de jeu. 
```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```
Par défaut, le serveur écoute sur le port 8000.
Pour pouvoir y accéder à distance, il faut faire une redirection de port.
Sur sa propre machine (local), il faut lancer la commande suivante :
```bash
ssh -L 8000:localhost:8000 username@remote_host
```
où username est votre nom d'utilisateur (e.g. `thomas.gastellu`) et remote_host est le nom de la machine du BR (e.g. `ferrari.polytechnique.fr`).

Puis on peut accéder au serveur de jeu en ouvrant un navigateur et en allant sur `http://localhost:8000`.



