import pathlib
import pickle

import torch



def weighted_jaccard_from_adj(adj1, nodes1, adj2, nodes2):
    # Step 1: build common node set
    all_nodes = sorted(set(nodes1) | set(nodes2))
    idx1 = {node: i for i, node in enumerate(nodes1)}
    idx2 = {node: i for i, node in enumerate(nodes2)}
    n = len(all_nodes)

    # Step 2: expand adj matrices into common indexing
    A1 = torch.zeros((n, n), dtype=torch.float32)
    A2 = torch.zeros((n, n), dtype=torch.float32)
    print(f"start n={n}")
    for i, u in enumerate(all_nodes):
        print(i)
        for j, v in enumerate(all_nodes):
            if u in idx1 and v in idx1:
                A1[i, j] = abs(adj1[idx1[u], idx1[v]])
            if u in idx2 and v in idx2:
                A2[i, j] = abs(adj2[idx2[u], idx2[v]])

    # Step 3: compute weighted Jaccard
    num = torch.minimum(A1, A2).sum().item()
    den = torch.maximum(A1, A2).sum().item()
    return num / den if den != 0 else 0.0

def load_data(path):
    """Load trimmed_feature_dict and prompts_and_results from a file."""
    with open(path, 'rb') as f:
        return pickle.load(f)  # returns (trimmed_feature_dict, prompts_and_results)

comparions = [
    ("flowers", "step_by_step_flowers"),
    ("subtraction", "step_by_step_subtraction"),
    ("word_addtition", "step_by_step_word_addtition"),
 #  ("addition", "step_by_step_addition"),

    ("subtraction", "addition"),
    ("flowers", "addition"),
    ("word_addtition", "addition"),

    ("step_by_step_addition", "step_by_step_flowers"),
    ("step_by_step_addition", "step_by_step_word_addtition"),
    ("step_by_step_addition", "step_by_step_subtraction"),
    ]
DIR = pathlib.Path(r"C:\Users\user\Downloads\category_graphs")

topics = ['flowers',
 'step_by_step_flowers',
 'step_by_step_subtraction',
 'step_by_step_word_addtition',
 'subtraction',
 'word_addtition']
topic_feature_paths = {topic: pathlib.Path(DIR / f"feature_counter_{topic}.pkl")  for topic in topics}
topic_edges_path = {topic: f"edges_{topic}.pkl"  for topic in topics}

if __name__ == '__main__':
    resutls = []
    for t1, t2 in comparions:
        print(f"Comparing {t1} vs {t2}")
        if t1 == "step_by_step_addition":
            features_1 = load_data(pathlib.Path(r"C:\Users\user\Downloads\features__cot.pkl"))[1]
            edges_1 = load_data(pathlib.Path(r"C:\Users\user\Downloads\edges_final_cot.pkl"))[1]
        elif t1 == "addition":
            features_1 = load_data(pathlib.Path(r"C:\Users\user\Downloads\features__reg.pkl"))[1] # features[digit] = set of (layer, fn)
            edges_1 = load_data(pathlib.Path(r"C:\Users\user\Downloads\edges_final_reg.pkl"))[1]
        else:
            features_1 = load_data(topic_feature_paths[t1])
            edges_1 = load_data(topic_edges_path[t1])

        if t2 == "step_by_step_addition":
            features_2 = load_data(pathlib.Path(r"C:\Users\user\Downloads\features__cot.pkl"))[1]
            edges_2 = load_data(pathlib.Path(r"C:\Users\user\Downloads\edges_final_cot.pkl"))[1]
        elif t2 == "addition":
            features_2 = load_data(pathlib.Path(r"C:\Users\user\Downloads\features__reg.pkl"))[1] # features[digit] = set of (layer, fn)
            edges_2 = load_data(pathlib.Path(r"C:\Users\user\Downloads\edges_final_reg.pkl"))[1]
        else:
            features_2 = load_data(topic_feature_paths[t2])
            edges_2 = load_data(topic_edges_path[t2])


        a = weighted_jaccard_from_adj( edges_1,features_1, edges_2, features_2)
        resutls.append(a)
        print(f"result of {t1} vs {t2} is: {a}")
    print(resutls)