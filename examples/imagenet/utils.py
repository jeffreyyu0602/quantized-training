def _pair_conv_bn(layers):
    pairs = []
    layer_dict = {}

    # Organize layers by prefix
    for layer in layers:
        prefix = '.'.join(layer.split('.')[:-1])
        if prefix not in layer_dict:
            layer_dict[prefix] = []
        layer_dict[prefix].append(layer)

    # Find pairs of conv and bn
    for prefix, layer_list in layer_dict.items():
        if "downsample" in prefix:
            pairs.append([f"{prefix}.0", f"{prefix}.1"])
        else:
            conv_layers = sorted([l for l in layer_list if 'conv' in l])
            bn_layers = sorted([l for l in layer_list if 'bn' in l])

            # Pair each conv with its corresponding bn
            for conv, bn in zip(conv_layers, bn_layers):
                pairs.append([conv, bn])

    return pairs
