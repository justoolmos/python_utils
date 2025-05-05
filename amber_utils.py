import re
import os
import csv
from pathlib import Path
from collections import defaultdict
from typing import Union, List


def check_run(out_files):
    """
    Chequea que una simulacion haya terminado correctamente.
    Devuelve una lista de las simulaciones que no terminaron bien.

    """
    if type(out_files) == list:
        unfinished = []
        for i in out_files:
            with open(i,"r") as file:
                end = False
                for line in file.readlines():
                    if "R M S  F L U C T U A T I O N S" in line:
                        end = True
                if not end:
                    unfinished.append(i)

    else:
	with open(out_files,"r") as file:
            for line in file.readlines():
                if "R M S  F L U C T U A T I O N S" in line:
                        return "File ok"
            return "Run failed"

    return unfinished


def extract_energy_terms(files: List[str],
                         terms: List[str],
                         occurrences: Union[str, int] = "last",
                         output_csv: str = "energy_summary.csv"):
    """
    Extract specific energy terms from multiple text files and write them to a single CSV.

    Args:
        files (List[str]): List of file paths.
        terms (List[str]): List of energy term names to extract.
        occurrences (str|int): "first", "last", "all", or an integer for nth appearance.
        output_csv (str): Name of the output CSV file.
    """
    # create a dictionary of patterns for the terms we are searching
    pattern_dict = {
        term: re.compile(rf"{re.escape(term)}\s*=\s*([-\d\.Ee+]+)")
        for term in terms
    }

    rows = []

    for file_path in files:
        file_path = Path(file_path)
        # create a dictionary term:[values]
        matches = defaultdict(list)

        with open(file_path, "r") as f:
            for line in f:
                for term, pattern in pattern_dict.items():
                    match = pattern.search(line)
                    if match:
                        value = float(match.group(1))
                        matches[term].append(value)

        # Select the appropriate occurrences
        filtered = {}
        for term, values in matches.items():
            if occurrences == "first":
                filtered[term] = values[0] if values else None
            elif occurrences == "last":
                filtered[term] = values[-1] if values else None
            elif isinstance(occurrences, int):
                idx = occurrences - 1
                filtered[term] = values[idx] if len(values) > idx else None
            elif occurrences == "all":
                # Join all values as a string for CSV
                filtered[term] = ";".join(str(v) for v in values) if values else None
            else:
                raise ValueError(""""occurrences" must be 'first', 'last', 'all', or an integer.""")

        # Add filename info
        #filtered["filename"] = file_path.name
        rows.append(filtered)

    # Ensure all keys are present in every row
    headers = terms

    # Write the CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})

    print(f"Energy values written to: {output_csv}")

def get_header(imports, name = "run", n_tasks = 1, n_cpus = 1, partition = "gpu", exclude_nodes = [], other_sbatch = [], other_commands = []):
    """
    Devuelve un header para un trabajo de slurm.
    Parametros:
        -imports: lista de programas a importar, los posibles valores son:
             -amber: importa amber y ambertools
             -ani_amber: importa amber que corre ANI
             -gaussian: importa gaussian
             -conda=env: activa el entorno de conda de nombre "env"
        -name: nombre del trabajo
        -n_tasks: numero de tareas 
        -n_cpus: numero de cpu-cores que usa el trabajo por tarea
        -partition: el nombre de la partición
        -exclude_nodes: lista de nodos a excluir, soporta formato "nodo[a,b]"
        -other_sbatch: lista de otros términos #SBATCH a incluir, e.j. ["#SBATCH --nodelist=[]", "#SBATCH..."]
        -other_commands: otros comandos que se deseen agregar al header
    """
    header = ["#!/bin/bash \n"]
    header.append(f"#SBATCH --job-name={name}\n")
    header.append(f"#SBATCH --ntasks={n_tasks}\n")
    header.append(f"#SBATCH --cpus-per-task={n_cpus}\n")
    if partition == "gpu":
        header.append("""#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1\n""")
    else:
        header.append(f"#SBATCH -p {partition}\n")
    
    if len(exclude_nodes) > 0:
        n = ",".join(exclude_nodes)
        header.append(f"""#SBATCH --exclude={n}\n""")
    
    if len(other_sbatch) > 0:
        for i in other_sbatch:
            header.append(i+"\n")
    
    for j in imports:
        if j == "amber":
            header.append("source /home/jolmos/programas/amber24/amber24/amber.sh\n")
        
        elif j == "ani_amber":
            header.append("""source /home/jolmos/programas/amber22_final/amber22/amber.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jolmos/mambaforge/envs/ani-amber2/lib
""")
        elif j == "gaussian":
            header.append("source /home/jfolmos/programas/gaussian/setup_gauss.sh \n")
            
        elif "conda_env" in j:
            env = j.split("=")[-1]
            header.append(f"conda activate {env}\n")
    
    if len(other_commands) > 0:
        for k in other_commands:
            header.append(k+"\n")
    
    
    return "".join(header)
        

def batch_run_slurm(ind, n, header, template_command):
    """
    Toma una lista de indices "ind", un número de corridas n, un diccionario de
    parámetros para la función header y un template para el comando. 
    Crea carpetas run_n con un script run.sh adentro listo para correr el comando.
    "template_command" es debe tener formato de fstring con {} en las posiciones donde va el indice.
    """
    l = len(ind)
    l_per_f = l//n 
    i = 0
    for j in range(n):
        os.system(f"mkdir run{j}")
        with open(f"run{j}/run.sh","w") as file:
            file.write(header)
            for k in range(l_per_f):
                file.write(template_command.format(ind[i])+"\n")
                i += 1
    
    f = 0
    while i < l:
        with open(f"run{f}/run.sh","a") as file:
            file.write(template_command.format(ind[i])+"\n")
            i += 1
            f += 1
    
    return 
                
                
                
        
    
    
        
    
    
    
    
    
    
