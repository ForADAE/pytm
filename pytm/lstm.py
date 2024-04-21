import dgl.backend as F
import numpy as np

lstm_plan = []
lstm_plan_idx = -1
lstm_plan_log2 = []

degree_node_buckets = {}


def _bucketing(val):
    """Internal function to create groups on the values.

    Parameters
    ----------
    val : Tensor
        Value tensor.

    Returns
    -------
    unique_val : Tensor
        Unique values.
    bucketor : callable[Tensor -> list[Tensor]]
        A bucketing function that splits the given tensor data as the same
        way of how the :attr:`val` tensor is grouped.
    """
    sorted_val, idx = F.sort_1d(val)
    unique_val = F.asnumpy(F.unique(sorted_val))
    bkt_idx = []
    for v in unique_val:
        eqidx = F.nonzero_1d(F.equal(sorted_val, v))
        bkt_idx.append(F.gather_row(idx, eqidx))

    def bucketor(data):
        bkts = [F.gather_row(data, idx) for idx in bkt_idx]
        return bkts

    return unique_val, bucketor


def degree_buckets(graph):
    """Return a dictionary where the keys are degrees and the values are the corresponding node buckets.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.

    Returns
    -------
    dict
        A dictionary where the keys are degrees and the values are the corresponding node buckets.
    """
    degs = graph.in_degrees()
    nodes = graph.dstnodes()

    global degree_node_buckets

    # degree bucketing
    unique_degs, bucketor = _bucketing(degs)
    degree_node_buckets = {}
    for deg, node_bkt in zip(unique_degs, bucketor(nodes)):
        degree_node_buckets[deg] = node_bkt


def init_lstm_plan(graph, num_layers, swap_bits):
    """Initialize the LSTM plan to zero

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    """
    global lstm_plan
    global degree_node_buckets
    # print("==================================")
    # print(degree_node_buckets)
    # print("==================================")
    # print(len(degree_node_buckets))
    # print("==================================")

    lstm_plan.clear()

    for i in range(len(degree_node_buckets)):
        if (swap_bits & 1):
            lstm_plan.append(True)
        else:
            lstm_plan.append(False)
        swap_bits = swap_bits >> 1

    # lstm_plan = [0] * (len(degree_node_buckets) * num_layers - 1)


def init_lstm_plan_log2(graph, num_layers, swap_bits):
    """Initialize the LSTM plan to zero

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    """
    global lstm_plan_log2
    global degree_node_buckets

    log_deg = int(np.floor(np.log2(len(degree_node_buckets))))
    # lstm_plan_log2 = [0] * (log_deg * num_layers - 1)
    lstm_plan_log2.clear()

    for i in range((log_deg * num_layers - 1)):
        if (swap_bits & 1):
            lstm_plan_log2.append(True)
        else:
            lstm_plan_log2.append(False)
        swap_bits = swap_bits >> 1


def sync_lstm_plan():
    """Sync the LSTM plan to all workers"""
    global lstm_plan
    global lstm_plan_log2
    global degree_node_buckets

    lstm_plan = []
    # use lstm_plan_log2 to modify lstm_plan
    for i in range(len(degree_node_buckets)):
        index = int(np.floor(np.log2(i + 1)))
        lstm_plan.append(lstm_plan_log2[index])


def init_lstm_manager(graph, num_layers, swap_bits):
    global lstm_plan
    global lstm_plan_log2
    global degree_node_buckets

    degree_buckets(graph)

    # init_lstm_plan(graph, num_layers, swap_bits)
    init_lstm_plan_log2(graph, num_layers, swap_bits)
    sync_lstm_plan()

    global lstm_plan_idx
    lstm_plan_idx = -1


def lstm_swap():
    global lstm_plan
    global lstm_plan_idx

    lstm_plan_idx += 1

    if lstm_plan_idx >= len(lstm_plan):
        raise IndexError("Index out of range")

    if lstm_plan[lstm_plan_idx] == 0:
        return False
    else:
        return True

def get_lstm_plan():
    global lstm_plan
    global lstm_plan_idx
    global lstm_plan_log2
    return lstm_plan, lstm_plan_idx, lstm_plan_log2