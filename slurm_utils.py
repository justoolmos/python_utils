import os
import time
import subprocess


def get_header(imports, name = "run", cluster = "qb", n_tasks = 1, n_cpus = 1, partition = None, exclude_nodes = [], other_sbatch = [], other_commands = []):
    """
    Devuelve un header para un trabajo de slurm.
    Parametros:
        -imports: lista de programas a importar, los posibles valores son:
             -amber: importa amber y ambertools
             -ani_amber: importa amber que corre ANI
             -gaussian: importa gaussian
             -conda=env: activa el entorno de conda de nombre "env"
        -name: nombre del trabajo
        -cluster: en que cluster se corre el trabajo:
            -"qb": cluster qb
            -"hg": hipergator
        -n_tasks: numero de tareas 
        -n_cpus: numero de cpu-cores que usa el trabajo por tarea
        -partition: el nombre de la partición:
            -None: default, asigna la partición default (generalmente cpu)
            -gpu: nodos asignados a la particion de gpu
        -exclude_nodes: lista de nodos a excluir, soporta formato "nodo[a,b]"
        -other_sbatch: lista de otros términos #SBATCH a incluir, e.j. ["#SBATCH --nodelist=[]", "#SBATCH..."]
        -other_commands: otros comandos que se deseen agregar al header
    """
    header = ["#!/bin/bash \n"]
    header.append(f"#SBATCH --job-name={name}\n")
    header.append(f"#SBATCH --ntasks={n_tasks}\n")
    header.append(f"#SBATCH --cpus-per-task={n_cpus}\n")
    if partition == "gpu" and cluster == "qb":
        header.append("""#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1\n""")
    
    elif partition == "gpu" and cluster == "hg":
        header.append("""#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --qos=mingjieliu-faimm
#SBATCH --account=mingjieliu-faimm\n""")

    else:
        header.append(f"#SBATCH -p {partition}\n")
    
    if len(exclude_nodes) > 0:
        n = ",".join(exclude_nodes)
        header.append(f"""#SBATCH --exclude={n}\n""")
    
    if len(other_sbatch) > 0:
        for i in other_sbatch:
            header.append(i+"\n")
    
    for j in imports:
        if cluster == "qb":
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
        
        elif cluster == "hg":
            if j == "amber":
                header.append("""
    module load gcc/12.2.0  openmpi/4.1.6  cuda/12.4.1

    module load amber/24
    """)
            elif j == "ani_amber":       
                raise ValueError("ANI-amber no disponible en hipergator")        

            elif j == "gaussian":
                header.append("module load gaussian/09 \n")

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

def wait_jobs(names = [], user = "jolmos", check_interval = 10 ):
    """
    Detiene la ejecución de un script hasta que termina un trabajo de slurm

    Parameters:
    -names: si se indica el nombre o lista de nombres se espera a que termine este
    -user: si no se indica el nombre se espera a que terminen todos los trabajos del usuario indicado
    -check_interval: tiempo de espera entre chequeos
    """
    if type(names) != list:
        names == [names]

    while True:
        end = True
        if len(names) > 0:
            for name in names:
                result = subprocess.run(
                    ["squeue", "-n", name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if name in result.stdout:
                    end = False
        else:
            result = subprocess.run(
                ["squeue", "-u", user],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if user in result.stdout:
                end = False
        if end:
            return
        else:
            time.sleep(check_interval)