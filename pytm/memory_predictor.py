import pandas as pd
import joblib
from torch.cuda import mem_get_info

def pre_gcn(num_nodes, num_edges, n_features, n_class, plan, model_path):
    model = joblib.load(model_path)
    test_row = pd.DataFrame({
        'num_nodes': [num_nodes],
        'num_edges': [num_edges],
        'n_features': [n_features],
        'n_class': [n_class],
        'plan': [plan]
    })
    return model.predict(test_row)

# return in a tuple (free, total)
def device_memory_info(dev_id=0):
    return mem_get_info(dev_id)

def gen_swap_plan(num_nodes, num_edges, n_features, n_class, model_path, safe_threshold=0.8):
    free, total = device_memory_info()
    # print(free, total)
    mem_ideal = float(total) * float(safe_threshold)
    for i in range(16):
        if pre_gcn(num_nodes, num_edges, n_features, n_class, i, model_path)[0][0] < mem_ideal:
            return i
    return 15