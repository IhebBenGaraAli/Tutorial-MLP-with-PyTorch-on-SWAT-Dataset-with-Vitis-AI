from pynq_dpu import DpuOverlay
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[2]:


# Charger l'overlay du DPU
overlay = DpuOverlay("dpu.bit")
overlay.load_model("_MLP_int.xmodel.xmodel")  # Charger le modèle


# In[3]:


dpu = overlay.runner  # Récupérer le runner


# In[4]:


# Charger la dataset
file_path = "SWaT_Dataset_Attack_v0.csv"
df = pd.read_csv(file_path)


# In[5]:


# Séparation des features (X) et des labels (y)
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels


# In[6]:


# Encoder les labels: "Normal" -> 0, "Attack" -> 1
y = np.where(y == "Normal", 0, 1)


# In[7]:


# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[8]:


# Sélection de 50000 échantillons "Normal" et 50000 "Attack"
normal_samples = X[y == 0][:50000]
attack_samples = X[y == 1][:50000]
test_samples = np.vstack((normal_samples, attack_samples)).astype(np.float32)


# In[9]:


# Récupérer les tenseurs d'entrée et de sortie du DPU
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()


# In[10]:


shapeIn = tuple(inputTensors[0].dims)  # Forme des entrées
shapeOut = tuple(outputTensors[0].dims)  # Forme des sorties
batch_size = shapeIn[0]  # Taille du batch du DPU


# In[11]:


# Préparer les buffers d'entrée et de sortie
input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]


# In[12]:


# Stocker les résultats de prédiction
predictions = []


# In[13]:


# Exécuter l'inférence batch par batch
for i in range(0, test_samples.shape[0], batch_size):
    batch = test_samples[i : i + batch_size]

    # S'assurer que la taille correspond au tenseur du DPU
    if batch.shape[0] < batch_size:
        pad = np.zeros((batch_size - batch.shape[0], batch.shape[1]), dtype=np.float32)
        batch = np.vstack((batch, pad))  # Ajouter du padding si nécessaire

    # Copier les données dans le buffer d'entrée
    input_data[0][:] = batch.reshape(shapeIn)

    # Lancer l'inférence
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    # Récupérer les résultats
    result = output_data[0][: batch.shape[0]]  # Enlever le padding éventuel
    predictions.append(np.argmax(result, axis=1))


# In[14]:


# Convertir la liste en un tableau numpy
predicted = np.concatenate(predictions, axis=0)[: test_samples.shape[0]]  # Supprimer le padding


# In[16]:


# Calculer l'accuracy manuellement
accuracy = np.mean(y[:100000] == predicted)

# Afficher la précision en pourcentage
print(f"Précision: {accuracy * 100:.2f}%")


# In[17]:


del dpu


# In[18]:


del overlay


# In[ ]:



