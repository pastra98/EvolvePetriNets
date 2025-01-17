# %%
import pickle
import sys
from collections import defaultdict
import gc
import os

def get_size(obj, seen=None):
    """Recursively calculate size of object and its contents in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(i, seen) for i in obj)
        
    return size

def analyze_pickle(filepath):
    """Analyze memory usage of a pickle file"""
    # Get file size
    file_size = os.path.getsize(filepath)
    print(f"Pickle file size: {file_size / 1024:.2f} KB")
    
    # Load pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Analyze object structure
    def get_object_sizes(obj, path="root", seen=None):
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return {}
        seen.add(obj_id)
        
        sizes = {path: get_size(obj)}
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                sizes.update(get_object_sizes(v, f"{path}.{k}", seen))
        elif hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                sizes.update(get_object_sizes(v, f"{path}.{k}", seen))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                sizes.update(get_object_sizes(v, f"{path}[{i}]", seen))
                
        return sizes
    
    # Get sizes of all objects
    sizes = get_object_sizes(data)
    
    # Sort by size
    sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
    
    print("\nLargest objects:")
    for path, size in sorted_sizes[:10]:
        print(f"{path}: {size / 1024:.2f} KB")
    
    # Analyze types
    type_sizes = defaultdict(int)
    def get_type_sizes(obj, seen=None):
        if seen is None:
            seen = set()
            
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        
        type_sizes[type(obj).__name__] += get_size(obj)
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                get_type_sizes(v, seen)
        elif hasattr(obj, '__dict__'):
            get_type_sizes(obj.__dict__, seen)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                get_type_sizes(v, seen)
    
    get_type_sizes(data)
    
    print("\nSize by type:")
    sorted_types = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)
    for type_name, size in sorted_types[:10]:
        print(f"{type_name}: {size / 1024:.2f} KB")

# %%

filepath = "./scripts/pickled_genomes/some_genome.pkl"
analyze_pickle(filepath)

# %%
with open(filepath, 'rb') as f:
    genome = pickle.load(f)

# genome.pop_component_tracker.component_dict
del genome.pop_component_tracker
genome.pop_component_tracker = None

with open("./scripts/pickled_genomes/some_genome_mod.pkl", 'wb') as f:
    pickle.dump(genome, f)