#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

## Author: Daniele Bagni, AMD/Xilinx Inc
## date 26 May 2023

# Nom du modèle quantifié
CNN_MODEL="MLP_int.xmodel"

# Spécifier le chemin du fichier arch.json
ARCH="./arch.json"

compile() {
  # Vérifier si le fichier .xmodel existe
  if [ ! -f "./build/quantized/${CNN_MODEL}" ]; then
    echo "ERREUR : Le fichier './build/quantized/${CNN_MODEL}' n'existe pas."
    echo "Assurez-vous que l'étape de quantification a été exécutée correctement."
    exit 1
  fi

  # Compiler le modèle
  vai_c_xir \
    --xmodel      ./build/quantized/${CNN_MODEL} \
    --arch        ${ARCH} \
    --output_dir  ./build/compiled_${TARGET} \
    --net_name    ${TARGET}_${CNN_MODEL}
}

# Exécuter la compilation
compile 2>&1 | tee build/log/compile_$TARGET.log

echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"


