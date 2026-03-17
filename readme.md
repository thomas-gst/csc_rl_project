# TRUCS QUE JAI PTETE REUSSI A FAIRE (EN TOUT CAS CA RUNNAIT)
- Modèle transformer, avec embeddings
- Connecter sur plusieurs serveurs
- Setup du behavoriol cloning avec recolte de data puis entrainement du model sur l'heuristic
- entrainement du model en ppo contre un adversaire heuristic

# TRUCS A VERIFIER / FAIRE

- j'avais zappé dans mon behavorial cloning d'entrainé le critic head, donc faut recheck
- pendant mon RL ppo j'avais l'impression que j'arrivais pas vraiment a reset, faut revoir la manière dont est chargé le modele preentrainé


## tips 

- vs pouvez utiliser "top" comme commande pour traquer l'usage du cpu/les process

utile pour savoir si vous avez bien réussi a lancer plusieurs serveurs, si ils tournent tous, si ils sont pas surchargés

- vs pouvez utiliser "watch nvidia-smi" pour track l'usage du GPU

utile pour savoir si l'entrainement se lance bien, si vs pouvez encore augmenter le batch size etc...

- si vous utilisez vs-code pour vous connecter au ssh vous pouvez directemnet forward les ports depuis vscode c'est super pratique

- les logs de ray pour tensorboardsont dans votre dossier home (cd ~/rayjspquoi)