#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "Activate environment..."
# conda activate vitis-ai-pytorch  # Activer l'environnement Vitis AI si nécessaire

# Répertoires des données et des poids
DATA_DIR=./build/data/
WEIGHTS=./build/float  # Répertoire où les poids du modèle sont stockés
DATASET=swat_dataset  # Nom du dataset 
GPU_ID=0
QUANT_DIR=./build/quantized  # Répertoire pour les fichiers quantifiés
export PYTHONPATH=${PWD}:${PYTHONPATH}


mkdir -p ${QUANT_DIR}
mkdir -p ${WEIGHTS}

echo "Conducting Quantization..."

# etape 1 : Calibration (mode calib)
echo "Running calibration..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py \
    --model mlp \
    --resume ${WEIGHTS}/swat_mlp.pt \
    --data_root ${DATA_DIR}/${DATASET} \
    --quant_mode calib \
    --quant_dir ${QUANT_DIR} \
    --device cpu  

# etape 2 : Test de la quantification (mode test)
echo "Running quantization test..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py \
    --model mlp \
    --resume ${WEIGHTS}/swat_mlp.pt \
    --data_root ${DATA_DIR}/${DATASET} \
    --quant_mode test \
    --quant_dir ${QUANT_DIR} \
    --device cpu  

# etape 3 : Déploiement (mode deploy)
echo "Running deployment..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test.py \
    --model mlp \
    --resume ${WEIGHTS}/swat_mlp.pt \
    --data_root ${DATA_DIR}/${DATASET} \
    --quant_mode test \
    --quant_dir ${QUANT_DIR} \
    --deploy \
    --device cpu  

