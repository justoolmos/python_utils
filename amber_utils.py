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

############################ gaussian ################################################

neutral_mol_density_header = """%chk=calculation.chk
%nprocshared=12
%mem=8GB
#P wB97X/6-31G* SCF=Tight Density=Current Pop=None

Gaussian Input Generated from PDB

0 1
"""

def pdb_to_gau_in(pdb_file, output_file, gau_header):
    """
    
    """
    atoms = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append((element, x, y, z))

    with open(output_file, 'w') as f:
        f.write(gau_header)
        for atom in atoms:
            f.write(f"{atom[0]:<2} {atom[1]:12.6f} {atom[2]:12.6f} {atom[3]:12.6f}\n")
        f.write("\n")
    
    return



    