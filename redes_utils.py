import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import random as rnd
import copy
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
import multiprocessing as mp

############################################### Utilidades ################################################################

def read_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def padding(in_array, len, axis = 0):
    """
    Agrega padding a un array a lo largo de un dado eje. El padding consiste de ceros.

    Parameters:
        -in_array: el array al que se le agrega el padding
        -len: el largo deseado 
        -axis: a lo largo de que eje se agrega el padding
    Returns:
        -array: array con el padding
    """
    current_len = in_array.shape[axis]
    if current_len >= len:
        return in_array  

    pad_width = [(0, 0)] * in_array.ndim
    
    right_pad = (len-current_len)//2
    left_pad = (len-current_len)-right_pad

    pad_width[axis] = (left_pad, right_pad)

    return np.pad(in_array, pad_width, mode='constant', constant_values=0)


############################################# Plots #######################################################################


def plot_ROC(y_true, y_predict, labels):
    """
    Toma dos listas, una de arrays de valores de referencia y una de arrays de 
    valores predichos. Genera una curva ROC por cada par referencia/predichos. 

    Parameters:
        - y_true: lista de los arrays de valores referencia
        - y_predict: lista de los arrays de valores predichos
        - labels: lista de labels para cada par y_true y_predict
    
    Returns:
        - una lista de tuplas (fpr, tpr, thresholds) para cada modelo y una lista de auc para cada curva.
        - genera el gráfico también
    """
    pares = list(zip(y_true, y_predict))
    plots = [roc_curve(i, j) for i,j in pares]   #la lista tiene grupos (fpr, tpr, thresholds) 
    auc = [roc_auc_score(i, j) for i,j in pares]
    
    plt.figure()
    for i in range(len(plots)):
        plt.plot(plots[i][0], plots[i][1], label=f'{labels[i]} (AUC = {auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    
    return plots, auc


def plot_train_v_valid(losses, nets, step, valid_in, valid_out, lf = nn.BCELoss()):
    """
    Grafica el error de entrenamiento y el de validación

    Parameters:
        -losses: lista/array de los errores de entrenamiento
        -nets: lista de redes
        -step: con que frecuencia se guardaron las redes y el error durante el entrenamiento
        -valid_in: tensor input de validación
        -valid_out: tensor output de validación
        -lf: loss function

    """
    losses_valid = [lf(nets[i].to("cpu")(valid_in),valid_out).item() for i in range(len(nets))]
    plt.plot(np.array(range(len(nets)))*step, losses, label = "train error")
    plt.plot(np.array(range(len(nets)))*step, losses_valid, label="validation error")
    plt.ylabel("BCE")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    return 


###################################### Funciones para entrenar #############################################################

def train_net(net, lf, optim, data_loader, epochs, device, save_freq):
    """
    Entrena un dado modelo por el número de epocas indicado.
    Devuelve dos listas, una con la loss y otra con la red cada 100 pasos de entrenamiento
    """
    loss_prog = []
    net_prog = []
    net = net.to(device)
    net.train()
    for epoch in range(epochs):
        for in_data, out_data in list(data_loader):
            if device != None:
                in_data = in_data.to(device)
                out_data = out_data.to(device)
            optim.zero_grad() 
            predict = net(in_data)
            loss = lf(predict,out_data)
            loss.backward()
            optim.step()
        if epoch % save_freq == 0:
            loss_prog.append(float(loss))
            net_prog.append(copy.deepcopy(net))
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    net.eval()
    return loss_prog, net_prog

def train_model_on_gpu(gpu_id, model, train_in, train_out, b_size = 200, n_epochs=30000, lf=nn.BCELoss(), l_r=0.0001, label = "net", save_freq=100, opt="SGD"):
    """
    Entrena un modelo en gpu.
    Parameters:
        - gpu_id: id de la gpu a usar
        - model: modelo a entrenar
        - train_in: tensor de datos input para entrenamiento
        - train_out: tensor de datos output para entrenamiento
        - b_size: tamaño del batch
        - n_epochs: cantidad de epocas a entrenar
        - lf: funcion de perdida a usar, objeto de la clase nn.Loss
        - l_r: learning rate
        - label: nombre con el que se guardan los resultados
        - save_freq: cada cuantas epocas se guardan los resultados

    Returns:
    genera dos archivos en el directorio actual:
        - {label}_losses.pkl: contiene la lista de perdidas por epoca
        - {label}_nets.pkl: contiene la lista de redes entrenadas cada "save_freq" epocas
    """    
    torch.cuda.set_device(gpu_id)

    # Move data to GPU
    train_in = train_in.to(f"cuda:{gpu_id}")
    train_out = train_out.to(f"cuda:{gpu_id}")

    dataset = TensorDataset(train_in, train_out)
    loader = DataLoader(dataset, batch_size=b_size, shuffle=True)

    # Loss and optimizer
    loss_fn = lf
    if opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=l_r)

    elif opt== "Adam":
        optimizer = optim.Adam(model.parameters(), lr=l_r)
    
    else:
        raise ValueError("Optimizer not recognized. Use 'SGD' or 'Adam'.")
    
    loss_history, model_snapshots = train_net(model, loss_fn, optimizer, loader, n_epochs, f"cuda:{gpu_id}", save_freq)

    print(f"Training on GPU {gpu_id} completed.")

    with open(f"{label}_losses.pkl", "wb") as f_loss:
        pickle.dump(loss_history, f_loss)

    with open(f"{label}_nets.pkl", "wb") as f_net:
        pickle.dump(model_snapshots, f_net)
    
    train_in.to("cpu")
    train_out.to("cpu")
    model.to("cpu")

    return


def run_trains_parallel(models_params):
    """
    Runs several training processes in paralle, each on a different GPU. Its important to have run "multiprocessing.set_start_method("spawn")"
    
    Parameters:
    - models_params: list of dictionariess, each one contains all the parameters for running each mode
                     according to "train model on gpu" function, except for gpu_id. They are passed as *args.
    
    Returns:
        -losses, models: both are lists of lists, each one contains the losses and models for each training
    """
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs.")
    
    if len(models_params) > num_gpus:
        raise ValueError("More models than GPUs available.")
    
    processes = []
    for i in range(len(models_params)):
        p = mp.Process(target=train_model_on_gpu, args=(i, *models_params[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

############################################### Modulos especializados #################################################

class CNN1d_to_Linear(nn.Module):
    """
    Esta clase define dos capas, una convolucional seguida de ReLU y una fully connected
    """
    def __init__(self, out_channel_num, kernel_len, linear_out_len, nt_len):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=out_channel_num, kernel_size=kernel_len) #La capa es una convolución 1D
        conv_output_size = out_channel_num * ( nt_len - kernel_len + 1) # (L - K + 1) formula para el tamaño del canal de salida dado el largo del canal de entrada y el kernel
        self.fc1 = nn.Linear(conv_output_size, linear_out_len)
    
    def forward(self, x):
        x = F.relu(self.conv1(x)) #Aplicamos ReLU a la salida de la convolucional
        x = x.view(x.size(0), -1) #Reshape de la salida a una matriz (n_seqs, flatten salida de la convolucional)
        x = self.fc1(x) #paso la salida por la capa lineal
        return x


