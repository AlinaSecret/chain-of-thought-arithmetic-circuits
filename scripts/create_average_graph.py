import io
import pathlib
import pickle

import torch
from pyexpat import features

import circuit_tracer


DIR = pathlib.Path(r"C:\Users\user\Downloads\category_graphs")

IS_COT = True

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            # This is the function that gets called to load tensors
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_data(path):
    with open(path, "rb") as f:
        # Use our custom unpickler to ensure data is mapped to the CPU
        return CPU_Unpickler(f).load()

def load_graph(i: int, topic: str):
    if IS_COT:
        file_extension = "_cot"
    else:
        file_extension = "_reg"
    return load_data( DIR /f"graphs_{i}_{topic}_{file_extension}.pkl")



def save_features(path, feature_to_effect):
    """Save trimmed_feature_dict and prompts_and_results to a file."""
    with open(path, 'wb') as f:
        pickle.dump(feature_to_effect, f)

FEATURES_PATH = pathlib.Path(r"C:\Users\user\Downloads\features__reg.pkl")

topics = ['flowers',
 'step_by_step_flowers',
 'step_by_step_subtraction',
 'step_by_step_word_addtition',
 'subtraction',
 'word_addtition']


if __name__ == '__main__':
    for topic in topics:
        print(f"topic: {topic}")
        features = load_data(pathlib.Path(DIR / f"feature_counter_{topic}.pkl"))  # features[digit] = set of (layer, fn)
        NUM_GRAPHS = 30

        # build mappings for each digit
        mapping = {feat: idx for idx, feat in enumerate(features)}

        # adjacency tensors: one per digit

        n = len(features)
        edges = torch.zeros((n, n), dtype=torch.float32)

        # main loop

        for graph_index in range(NUM_GRAPHS):
            print("graph_index", graph_index)
            graph = load_graph(graph_index, topic)

            circut_nodes = []
            for i, f in enumerate(graph.selected_features):
                layer, pos, fn = graph.active_features[f]

                if torch.is_tensor(layer):
                    layer = int(layer.item())
                else:
                    layer = int(layer)

                if torch.is_tensor(fn):
                    fn = int(fn.item())
                else:
                    fn = int(fn)

                if pos >= 15 and (layer, fn) in features:
                    circut_nodes.append((i, (layer, fn)))

            # accumulate edges into tensor
            for i, target in circut_nodes:
                tgt_idx = mapping[target]
                for j, source in circut_nodes:
                    src_idx = mapping[source]
                    edges[src_idx, tgt_idx] += graph.adjacency_matrix[i][j]

            save_features(f"edges_{topic}.pkl", edges)
            print(f"Progress saved after {topic}, graph {graph_index}")

        # final save
        save_features(f"edges_final_{topic}.pkl", edges)
        print("Final edges saved.")