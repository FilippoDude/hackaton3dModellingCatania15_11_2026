import json
import os
import numpy as np
import trimesh

# Definizione dei percorsi delle cartelle
INPUT_DIR = "inputObj"
INTERMEDIATE_DIR = "intermediate"
OUTPUT_DIR = "outputObj"

def setup_directories():
    """Crea le cartelle necessarie se non esistono."""
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_obj_with_transform(obj_filename, matrix_world):
    """Carica un file OBJ dalla cartella inputObj e applica la matrice di trasformazione 4x4."""
    obj_path = os.path.join(INPUT_DIR, obj_filename)
    mesh = trimesh.load(obj_path, force='mesh')
    
    matrix = np.array(matrix_world)
    mesh.apply_transform(matrix)
    
    return mesh

def convert_mesh_to_mit(mesh, grid_resolution=128):
    """
    Converte una mesh tridimensionale nella struttura dati MIT (Marching Intersections).
    """
    bounds = mesh.bounds
    min_bound, max_bound = bounds[0], bounds[1]
    
    mit_grid = {
        "resolution": grid_resolution,
        "bounds": [min_bound.tolist(), max_bound.tolist()],
        "intersections": {} 
    }
    
    ray_origins = []
    ray_directions = []
    
    x_coords = np.linspace(min_bound[0], max_bound[0], grid_resolution)
    y_coords = np.linspace(min_bound[1], max_bound[1], grid_resolution)
    
    for x in x_coords:
        for y in y_coords:
            ray_origins.append([x, y, min_bound[2] - 1.0])
            ray_directions.append([0.0, 0.0, 1.0])
            
    ray_origins = np.array(ray_origins)
    ray_directions = np.array(ray_directions)
    
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True
    )
    
    for ray_idx, loc in zip(index_ray, locations):
        if ray_idx not in mit_grid["intersections"]:
            mit_grid["intersections"][int(ray_idx)] = []
        mit_grid["intersections"][int(ray_idx)].append(float(loc[2]))
        
    for ray_idx in mit_grid["intersections"]:
        mit_grid["intersections"][ray_idx].sort()
        
    return mit_grid

def save_mit_file(mit_data, filename):
    """Salva la struttura MIT (in formato JSON) nella cartella intermediate."""
    filepath = os.path.join(INTERMEDIATE_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(mit_data, f)
    print(f"File MIT salvato in: {filepath}")

def perform_boolean_operation(target_mesh, tool_mesh, operation_type):
    """
    Esegue l'operazione booleana (UNION, DIFFERENCE, INTERSECTION) 
    sulle mesh e restituisce la mesh risultante.
    """
    op = operation_type.upper()
    
    try:
        if op == "UNION":
            result_mesh = target_mesh.union(tool_mesh)
        elif op == "DIFFERENCE" or op == "SUBTRACT":
            result_mesh = target_mesh.difference(tool_mesh)
        elif op == "INTERSECTION":
            result_mesh = target_mesh.intersection(tool_mesh)
        else:
            print(f"Operazione {op} non riconosciuta, eseguo UNION di default.")
            result_mesh = target_mesh.union(tool_mesh)
            
        return result_mesh
    except Exception as e:
        print(f"Errore durante l'esecuzione dell'operazione booleana: {e}")
        return None

def process_operations_json(json_path):
    """Legge il file JSON delle operazioni ed elabora target e tool."""
    setup_directories()
    
    with open(json_path, 'r') as f:
        operations = json.load(f)
        
    for op in operations:
        op_id = op["id"]
        op_type = op["operation"]
        
        print(f"\n--- Elaborazione operazione #{op_id}: {op_type} ---")
        
        # 1. Caricamento Target
        target_data = op["target"]
        target_mesh = load_obj_with_transform(target_data["obj_snapshot"], target_data["matrix_world"])
        
        # 2. Caricamento Tool
        tool_data = op["tool"]
        tool_mesh = load_obj_with_transform(tool_data["obj_snapshot"], tool_data["matrix_world"])
        
        # 3. Conversione in formato MIT (Marching Intersections)
        print("Conversione Target in MIT...")
        target_mit = convert_mesh_to_mit(target_mesh)
        target_mit_filename = target_data["obj_snapshot"].replace(".obj", ".mit")
        save_mit_file(target_mit, target_mit_filename)
        
        print("Conversione Tool in MIT...")
        tool_mit = convert_mesh_to_mit(tool_mesh)
        tool_mit_filename = tool_data["obj_snapshot"].replace(".obj", ".mit")
        save_mit_file(tool_mit, tool_mit_filename)
        
        # 4. Esecuzione dell'operazione booleana e salvataggio del risultato in outputObj/
        print(f"Esecuzione operazione booleana ({op_type})...")
        result_mesh = perform_boolean_operation(target_mesh, tool_mesh, op_type)
        
        output_filename = f"{str(op_id).zfill(6)}_result.obj"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        
        if result_mesh and not result_mesh.is_empty:
            result_mesh.export(output_filepath)
            print(f"Risultato salvato con successo in: {output_filepath}")
        else:
            print(f"Impossibile generare una mesh valida per l'operazione #{op_id}.")

if __name__ == "__main__":
    process_operations_json("operations.json")