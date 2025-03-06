import torch
import argparse
import os
import pandas as pd
import numpy as np
from pytorch_nndct.apis import torch_quantizer

# Définir la classe MLP
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, num_classes):
        super(MLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden1),
            torch.nn.BatchNorm1d(hidden1),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.BatchNorm1d(hidden2),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2, hidden3),
            torch.nn.BatchNorm1d(hidden3),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden3, num_classes)
        )

    def forward(self, x):
        return self.model(x)
        
def load_dataset(data_path):

    data = pd.read_csv(data_path)
    
    features = data.iloc[:, :51].values  # 51 
    labels = data.iloc[:, 51].values     # 1 
    
    # encoder les labels : "Normal" -> 0, "Attack" -> 1
    label_mapping = {"Normal": 0, "Attack": 1}
    labels = np.array([label_mapping[label] for label in labels])
    
    # Convertir en tenseurs PyTorch
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return features, labels

def main():
    # Arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Quantization and Testing for MLP Model")
    parser.add_argument('--model', type=str, default='mlp', help='Model type (mlp)')
    parser.add_argument('--resume', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--quant_mode', type=str, choices=['calib', 'test'], required=True, help='Quantization mode')
    parser.add_argument('--quant_dir', type=str, required=True, help='Directory for quantized files')
    parser.add_argument('--deploy', action='store_true', help='Enable deploy mode')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    args = parser.parse_args()

    # Forcer l'utilisation du CPU
    device = torch.device('cpu')  # Convertir en objet torch.device
    print(f"Using device: {device}")

    # Paramètres du modèle
    input_size = 51  # Nombre de features en entrée
    hidden1 = 256
    hidden2 = 128
    hidden3 = 64
    num_classes = 2  # Nombre de classes (Normal/Attack)

    # Charger le modèle
    model = MLP(input_size, hidden1, hidden2, hidden3, num_classes)
    model.load_state_dict(torch.load(args.resume, map_location=device))
    model.eval()

    # Charger le dataset 
    data_path = os.path.join(args.data_root, "SWaT_Dataset_Attack_v0.csv")
    features, labels = load_dataset(data_path)


    # Quantizer
    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(features[0].unsqueeze(0).to(device),), 
        output_dir=args.quant_dir,
        device=device  
    )

    # Mode calibration
    if args.quant_mode == 'calib':
        print("Running calibration...")
        for i in range(1000): 
            input_data = features[i].unsqueeze(0).to(device)  
            quant_model = quantizer.quant_model(input_data)
        quantizer.export_quant_config()  # Exporter la configuration de quantification

      # Mode test
    elif args.quant_mode == 'test':
        print("Running quantization test...")
        for i in range(1000):  
            input_data = features[i].unsqueeze(0).to(device)  
            quant_model = quantizer.quant_model(input_data)
            print(f"Quantized model output for sample {i}: {quant_model}")

    # Mode déploiement
    if args.deploy:
        print("Exporting quantized model for deployment...")
        quantizer.export_quant_config()
        quantizer.export_xmodel(output_dir=args.quant_dir,deploy_check=True)  

if __name__ == '__main__':
    main()

