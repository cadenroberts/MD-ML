
def load_state_dict_with_rename(model, state_dict):
    """Map between the parameter names used in TorchMD 2.0.0 and 2.4.0"""

    renamed_dict = state_dict.copy()
    keys_2_0_0 = [
        "output_model.output_network.0.weight",
        "output_model.output_network.0.bias",
        "output_model.output_network.2.weight",
        "output_model.output_network.2.bias",
    ]
    keys_2_4_0 = [
        "output_model.output_network.layers.0.weight",
        "output_model.output_network.layers.0.bias",
        "output_model.output_network.layers.2.weight",
        "output_model.output_network.layers.2.bias",
    ]
    if keys_2_0_0[0] in model.state_dict().keys():
        remap = [keys_2_4_0, keys_2_0_0]
    else:
        remap = [keys_2_0_0, keys_2_4_0]

    for a, b in zip(*remap):
        if a in renamed_dict:
            renamed_dict[b] = renamed_dict[a]
            del renamed_dict[a]

    model.load_state_dict(renamed_dict)