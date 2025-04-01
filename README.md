# Tutoriel : MLP avec PyTorch sur le Dataset SWAT et Vitis AI

## 📌 Introduction

Ce tutoriel vous guide à travers les étapes nécessaires pour entraîner, quantifier et déployer un modèle **MLP (Multi-Layer Perceptron)** avec **PyTorch** sur le dataset **SWAT**, en utilisant **Vitis AI** pour l'optimisation et l'exécution sur un **DPU (Deep Learning Processing Unit)**. Vous apprendrez à :

1. Entraîner un modèle MLP avec PyTorch. 
2. Installer et configurer Vitis AI.
3. Quantifier le modèle pour une exécution optimisée sur FPGA.
4. Déployer et exécuter le modèle sur une carte cible.
5. Mesurer la consommation énergétique pendant l'inférence.

---

## 📥 Installation de Vitis AI

### Clonage du dépôt Vitis AI

Commencez par cloner le dépôt officiel de Vitis AI :

```sh
git clone https://github.com/Xilinx/Vitis-AI
```

### Configuration de l'environnement

Dans ce tutoriel, nous supposons que vous avez installé **Vitis AI 3.5** (abrégé VAI3.5) dans votre système de fichiers. Définissez le répertoire de travail comme suit :

```sh
export WRK_DIR=/media/danieleb/DATA/VAI3.5
```

### Construction de l'image Docker

Depuis le répertoire **Vitis AI 3.5**, exécutez les commandes suivantes pour construire l'image Docker :

```sh
cd ${WRK_DIR}
cd docker
./docker_build.sh -t gpu -f pytorch
```

Une fois terminé, vérifiez l'image Docker créée :

```sh
docker images
```

Vous devriez voir une sortie similaire à :

```
REPOSITORY                        TAG               IMAGE ID       CREATED         SIZE
xilinx/vitis-ai-pytorch-gpu   3.5.0.001-b56bcce50   3c5d174a1807   27 hours ago    21.4GB
```

---

## 📂 Répertoire de travail

### Création du répertoire du tutoriel

Créez un dossier **tutorials** sous **${WRK_DIR}**, puis copiez ce tutoriel dans ce dossier et renommez-le en **Tutorial-MLP-with-PyTorch-on-SWAT-Dataset-with-Vitis-AI**.

Avec la commande suivante, vous pouvez vérifier la structure des dossiers :

```sh
tree -d -L 2
```

Vous devriez voir une structure similaire à :

```
${WRK_DIR} # votre répertoire de travail Vitis AI 3.5
.
├── bck
├── board_setup
│   ├── v70
│   └── vek280
├── demos
├── docker
│   ├── common
│   ├── conda
│   └── dockerfiles
├── docs
│   ├── docs
│   ├── _downloads
│   ├── doxygen
│   ├── _images
│   ├── _sources
│   └── _static
├── docsrc
│   ├── build
│   └── source
├── dpu
├── examples
│   ├── custom_operator
│   ├── ofa
│   ├── OnBoard
│   ├── vai_library
│   ├── vai_optimizer
│   ├── vai_profiler
│   ├── vai_quantizer
│   ├── vai_runtime
│   ├── waa
│   └── wego
├── model_zoo
│   ├── images
│   └── model-list
├── src
│   ├── AKS
│   ├── vai_library
│   ├── vai_optimizer
│   ├── vai_petalinux_recipes
│   ├── vai_quantizer
│   └── vai_runtime
├── third_party
│   ├── tflite
│   └── tvm
└── tutorials # créé par vous
    ├── Tutorial-MLP-with-PyTorch-on-SWAT-Dataset-with-Vitis-AI # ce tutoriel
```

---

## 🚀 Lancement du conteneur Docker

### Démarrage du conteneur

Depuis **${WRK_DIR}**, lancez le conteneur Docker Vitis AI :

```sh
cd ${WRK_DIR}
./docker_run.sh xilinx/vitis-ai-pytorch-gpu:latest
```

### Activation de l'environnement

Activez l'environnement Conda et accédez au répertoire du tutoriel :

```sh
conda activate vitis-ai-pytorch
cd /workspace/tutorials/Tutorial-MLP-with-PyTorch-on-SWAT-Dataset-with-Vitis-AI
```

💡 **Remarque** : Le dossier `/workspace` dans le conteneur est mappé au système de fichiers hôte.

---

## 📊 Dataset SWAT

### Téléchargement du dataset

Nous utilisons le dataset **SWaT_Dataset_Attack_v0.csv**, disponible ici :

🔗 [Télécharger le dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

### Caractéristiques du dataset

- **11 jours d'opérations** (7 jours normaux, 4 jours avec attaques).
- **51 capteurs et actionneurs** surveillés.
- **41 attaques simulées**.

### Placement du dataset

Le dataset doit être placé dans :

```sh
/build/data/swat_dataset
```

---

## 🎯 Entraînement du modèle MLP

### Exécution du script d'entraînement

Dans le répertoire **train_mlp**, exécutez le script **train_mlp_pytorch.ipynb** pour entraîner un MLP avec PyTorch.

- Le modèle atteint une **précision de 98%**.
- Les poids sont sauvegardés en **format `.pt`** dans **`build/float`**.

---

## 🔍 Quantification du modèle

### Exécution de la quantification

Exécutez le script de quantification :

```sh
cd /workspace/vitis_ai_mlp
./scripts/quant.sh
```

---

## 🛠 Compilation pour le DPU

### Adaptation du fichier `arch.json`

Adaptez le fichier **arch.json** pour correspondre au fingerprint du DPU cible.

### Compilation du modèle

Exécutez la compilation :

```sh
cd /workspace/vitis_ai_mlp
./comp.sh
```

---

## 🏃‍♂️ Exécution sur la carte cible

### Déploiement du modèle

Après la compilation, copiez le modèle **`.xmodel`** et le dataset dans **`inference_dpu_mlp`**, puis transférez-les sur la carte cible.

---

## ⚡ Mesure de la consommation

### Mesure de la consommation énergétique

Dans **`inference_dpu_mlp`**, utilisez le script **measure_power** pour mesurer la consommation énergétique durant l'inférence.

---

## 🏁 Conclusion

Ce tutoriel vous a guidé à travers tout le processus d'entraînement, quantification et exécution d'un modèle MLP avec **Vitis AI** sur le dataset **SWAT**. Vous avez également appris à mesurer la consommation énergétique pendant l'inférence.

💡 **Pour toute question, ouvrez une issue sur GitHub !** 🚀

---
