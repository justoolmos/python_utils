#secuencia con distribuciones de fondo

import random as rnd
import numpy as np

def fasta_to_strings(fasta_dir):
    """
    Toma un archivo .fasta y devuelve las secuencias del mismo en una lista
    """
    seqs = []
    with open(fasta_dir,"r") as file:
        seq = ""
        for line in file.readlines():
            if ">" in line:
                seqs.append(seq)
                seq = ""
            else:
                seq += line.strip("\n")
        seqs.append(seq)
        
    return seqs[1:]


def seq_to_onehot(seq, dim = 1):
    """
    Transforma una secuencia de ADN de largo N a un tensor de largo
    N*4 one hot encoded. Las posiciones para las bases son ACGT.
    """
    encoding = {"A":np.array([1,0,0,0]), "C":np.array([0,1,0,0]), "G":np.array([0,0,1,0]), "T":np.array([0,0,0,1]),"a":np.array([1,0,0,0]), "c":np.array([0,1,0,0]), "g":np.array([0,0,1,0]), "t":np.array([0,0,0,1])}
    array = np.zeros((len(seq),4))
    
    i = 0
    while i < len(seq):
        array[i,:] = encoding[seq[i]]
        i += 1
    
    if dim == 1:
        return array.flatten()
    
    elif dim == 2:
        return array.reshape((4,i))


def make_rand_seq(l):
    """
    Toma un valor de largo l y crea una secuencia al azar de ADN
    """
    bases = "ACGT"
    seq = "".join([rnd.choice(bases) for i in range(l)])
    
    return seq

    
def permutar_seq(seq,seg_size, n_perm):
    """

    Parameters
    ----------
    seq : string de la secuencia a permutar
    seg_size : tamaño de los segmentos que se permutan
    n_perm : número de permutaciones

    Returns
    -------
    secuencia permutada

    """
    segments = [seq[i:i+seg_size] for i in range(0, len(seq), seg_size)]
    segments_to_change = segments[:len(segments)-1]
    last_segment = segments[-1]
    
    for i in range(n_perm):
        ids = list(range(0,len(segments_to_change)))
        id1 = rnd.choice(ids)
        id2 = rnd.choice(ids)
        
        while id2 == id1:
            id2 = rnd.choice(ids)
            
        segments_to_change[id1], segments_to_change[id2] = segments_to_change[id2], segments_to_change[id1]
    
    return "".join(segments_to_change)+last_segment


def get_trans_matrix_HKY(freqs, alpha= 0.4483, beta=0.0082):
    """
    devuelve una matriz de 4x4 con las probabilidades de cambio de
    las bases según el modelo HKY
    
    Parameters
    ----------
    freqs: las frecuencias de las bases en orden ACGT
    alpha, beta:  parametros de transicion y transversion
    
    """
    A,C,G,T = freqs
    a = alpha
    b = beta
    
    array = np.array([[1-(a*G+b*T+b*C), b*C, a*G, b*T],
                      [b*A, 1-(a*T+b*A+b*G), b*G, a*T],
                      [a*A, b*C, 1-(a*A+b*C+b*T), b*T],
                      [b*A, a*C, b*G, 1-(a*C+b*G+b*A)]])
    
    return array

def get_base_freqs(seqs):
    """
    Calcula la frecuencia de cada base en una lista de secuencias.
    Devuelve las frequencias en una lista [A,C,G,T]
    """
    data = np.array([list(i) for i in seqs])
    vals, count = np.unique(data, return_counts = True)
    dic = dict(zip(vals,count))
    N = data.shape[0]*data.shape[1]
    
    freqs = []
    for i in ["A","C","G","T"]:
        freqs.append(float(dic[i] / N))
    
    return freqs
    
def mutate_seq(seq, n, matrix):
    """
    Realiza n mutaciones en posiciones aleatorias de la secuencia "seq" siguiendo
    una matriz de probabilidad de cambios. En cada paso se fuerza una mutacion, 
    osea, se ignora la probabilidad de mantener la base.
    
    Parameters
    ----------
    seq : string de la secuencia        
    n : numero de mutaciones
    matrix: la matriz de probabilidades de cambio
    
    Returns
    -------
    secuencia mutada
    """
    bases = ['A', 'C', 'G', 'T']
    base_to_idx = {b: i for i, b in enumerate(bases)}
    seq = list(seq)  # convert to list for mutation
    
    for t in range(n):
        pos = rnd.choice(range(len(seq)))
        original_base = seq[pos]
        i = base_to_idx[original_base]

        # Mutation probabilities, ignoring the original base
        probs = matrix[i].copy()
        probs[i] = 0
        probs = probs / probs.sum()  # renormalize

        new_base = np.random.choice(bases, p=probs)
        seq[pos] = new_base

    return ''.join(seq)
