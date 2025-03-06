# Tutoriel : Déploiement d'un Modèle MLP sur DPU avec PyTorch et Vitis AI

Ce tutoriel explique comment déployer un modèle de classification binaire (normal ou attaque) basé sur un MLP, entraîné avec PyTorch et optimisé pour l'inférence sur un DPU avec Vitis AI. Nous couvrons les étapes de quantification, compilation et évaluation des performances énergétiques sur la carte Zynq Ultra96.

## 1. Prérequis

- **Système Linux**
- **Docker installé et configuré**
- **Accès à Vitis AI 3.5**
- **Carte Zynq Ultra96 pour l'inférence sur DPU**

## 2. Installation de Vitis AI

Cloner le dépôt de Vitis AI avec :
```sh
git clone https://github.com/Xilinx/Vitis-AI
```

Depuis le répertoire Vitis AI 3.5, exécutez :
```sh
cd ${WRK_DIR}
cd docker
./docker_build.sh -t gpu -f pytorch
```
Vérifiez l'installation avec :
```sh
docker images
```

## 3. Lancer le conteneur Docker

Exécutez les commandes suivantes :
```sh
cd ${WRK_DIR}  # Répertoire de travail Vitis AI
./docker_run.sh xilinx/vitis-ai-pytorch-gpu:latest
conda activate vitis-ai-pytorch
cd /workspace/tutorials/Tutorial-MLP-with-PyTorch-on-SWAT-Dataset-with-Vitis-AI
```
Ajoutez les paquets nécessaires :
```sh
pip install randaugment torchsummary
```

## 4. Jeu de données SWaT

Le jeu de données SWaT est disponible ici :
[SWaT Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

Caractéristiques du fichier **SWaT_Dataset_Attack_v0.csv** :
- 11 jours d'opération (7 jours normaux, 4 jours avec attaques)
- Captures du trafic réseau et des 51 capteurs/actionneurs
- 41 attaques simulées et labellisées

Le fichier **SWaT_Dataset_Attack_v0.csv** doit être placé dans :
```
build/data/swat_dataset
```

## 5. Entraînement du modèle MLP

Dans le répertoire **train_mlp**, exécutez le script d'entraînement :
```sh
python train_mlp_pytorch.py
```
Ce script génère un modèle avec une précision de 98 %, sauvegardé sous forme de fichier `.pt`.

Les poids doivent être stockés dans :
```
build/float
```

## 6. Quantification du modèle

Exécutez la quantification avec :
```sh
/workspace/vitis_ai_mlp$ ./scripts/quant.sh
```

## 7. Compilation pour le DPU cible

Vérifiez que le fichier **arch.json** correspond à votre DPU, puis compilez :
```sh
./comp.sh
```

## 8. Inférence sur la carte Ultra96

Une fois la quantification et la compilation terminées :
- Copiez le modèle **.xmodel** dans `inference_dpu_mlp`
- Copiez tout le dossier sur la carte Ultra96
- Exécutez l'inférence sur le DPU

## 9. Mesure de la consommation énergétique

Dans le dossier **inference_dpu_mlp**, utilisez le script **measure_power** pour mesurer la consommation d'énergie pendant l'inférence.

## 10. Conclusion

Ce tutoriel vous permet de déployer un modèle MLP optimisé pour l'inférence sur DPU, en passant par l'entraînement, la quantification et la compilation. Vous pouvez ainsi mesurer les performances et la consommation énergétique du modèle sur la carte Ultra96.

🚀 Bon déploiement !

