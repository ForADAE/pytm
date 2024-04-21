import random
import string
import torch

managed_tensors = {}

swap_plan = []

swap_rate = 0.0


def set_swap_rate(rate):
    global swap_rate
    swap_rate = rate


def get_swap_rate():
    global swap_rate
    return swap_rate


def swap_decision():
    global swap_rate
    temp = random.random()
    print("swap_rate: ", swap_rate)
    print("temp: ", temp)
    if temp < swap_rate:
        return True
    else:
        return False


def clean_managed_tensors():
    managed_tensors.clear()


def gen_key():
    key = ''.join(
        random.choices(string.ascii_uppercase + string.ascii_lowercase +
                       string.digits,
                       k=20))
    while key in managed_tensors:
        key = ''.join(
            random.choices(string.ascii_uppercase + string.ascii_lowercase +
                           string.digits,
                           k=20))
    return key


def tensor_manage(t, key="__default__"):

    global swap_plan

    if key == "__default__":
        key = gen_key()
    if key in managed_tensors:
        key = gen_key()
        print(
            "[ERROR] tensor_manage: key already in managed_tensors, generate a new key: ",
            key)

    swap_flag = False
    if len(managed_tensors) < len(swap_plan):
        swap_flag = swap_plan[len(managed_tensors)]
    temp = {
        "tensor": t,
        "device": t.device,
        "swap_flag": swap_flag,
        "swap_lock": True
    }
    managed_tensors[key] = temp

    return key


def get_managed(key):
    if key not in managed_tensors:
        temp = {
            "tensor": torch.zeros(0),
            "device": torch.zeros(0).device,
            "swap_flag": False,
            "swap_lock": True
        }
        print("[ERROR] get_managed: key not in managed_tensors")
        return temp
    return managed_tensors[key]


def print_managed():
    for key in managed_tensors:
        print("key is:", key)
        print(managed_tensors[key])


def sync_managed_tensors():
    for key in managed_tensors:
        if managed_tensors[key]["swap_flag"] == True and managed_tensors[key][
                "swap_lock"] == False and managed_tensors[key][
                    "tensor"].device != managed_tensors[key]["device"]:
            managed_tensors[key]["tensor"].to_inplace(
                managed_tensors[key]["device"],
                dtype=managed_tensors[key]["tensor"].dtype)


def sync_managed_tensor(key):
    if key not in managed_tensors:
        print("[ERROR] sync_managed_tensor: key not in managed_tensors")
        return
    if managed_tensors[key]["swap_flag"] == True and managed_tensors[key][
            "swap_lock"] == False and managed_tensors[key][
                "tensor"].device != managed_tensors[key]["device"]:
        managed_tensors[key]["tensor"].to_inplace(
            managed_tensors[key]["device"],
            dtype=managed_tensors[key]["tensor"].dtype)


def unlock_managed_tensor(key):
    if key not in managed_tensors:
        print("[WARN ] unlock_managed_tensor: key not in managed_tensors ",
              key)
        return
    if managed_tensors[key]["swap_lock"] == False:
        return
    managed_tensors[key]["swap_lock"] = False
    if managed_tensors[key]["swap_flag"] == True:
        swap_to_cpu(key)


def swap_to_cpu(key):
    if key not in managed_tensors:
        print("[ERROR] swap_to_cpu: key not in managed_tensors")
        return
    if managed_tensors[key]["tensor"].device != torch.device("cpu"):
        managed_tensors[key]["tensor"].to_inplace(
            "cpu", dtype=managed_tensors[key]["tensor"].dtype)


def swap_to_device(key):
    if key not in managed_tensors:
        print("[ERROR] swap_to_device: key not in managed_tensors")
        return
    if managed_tensors[key]["tensor"].device != managed_tensors[key]["device"]:
        managed_tensors[key]["tensor"].to_inplace(
            managed_tensors[key]["device"],
            dtype=managed_tensors[key]["tensor"].dtype)


def get_swap_flags():
    flags = []
    for key in managed_tensors:
        flags.append(managed_tensors[key]["swap_flag"])
    return flags


def set_swap_flags(flags):
    if len(flags) != len(managed_tensors):
        print(
            "[ERROR] set_swap_flags: flags length not equal to managed_tensors length"
        )
        return
    for i, key in enumerate(managed_tensors):
        managed_tensors[key]["swap_flag"] = flags[i]


def get_swap_flag(key):
    if key not in managed_tensors:
        print("[ERROR] get_swap_flag: key not in managed_tensors")
        return False
    return managed_tensors[key]["swap_flag"]


def set_swap_flag(key, flag):
    if key not in managed_tensors:
        print("[ERROR] set_swap_flag: key not in managed_tensors")
        return
    managed_tensors[key]["swap_flag"] = flag


def set_swap_plan(swap_bits):
    global swap_plan
    # swap_bits is a integer, each bit represents whether to swap the corresponding tensor
    swap_plan.clear()
    while (swap_bits > 0):
        if (swap_bits & 1):
            swap_plan.append(True)
        else:
            swap_plan.append(False)
        swap_bits = swap_bits >> 1
