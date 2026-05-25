def get_model_name(model):
    model_repr = repr(model)
    class_name = model_repr.split("'")[1]
    model_name = class_name.split(".")[2]
    return model_name


def create_random_nonzero_mask(p_matrix):
    num_rows, num_cols = p_matrix.shape
    mask = torch.zeros_like(p_matrix, dtype=torch.bool)
    for col in range(num_cols):
        non_zero_indices = torch.nonzero(p_matrix[:, col] != 0, as_tuple=False)
        if non_zero_indices.numel() == 0:
            continue
        chosen_index = non_zero_indices[torch.randint(0, non_zero_indices.size(0), (1,))].item()
        mask[chosen_index, col] = True
    return mask



def random_mask_target_p(p, rate):
    target_p = p.clone()
    batch_size, n = target_p.shape
    for i in range(batch_size):
        non_zero_indices = (target_p[i] != 0).nonzero(as_tuple=False).squeeze()

        if len(non_zero_indices) > 0:
            num_to_zero = int(len(non_zero_indices) * rate)
            indices_to_zero = torch.randperm(len(non_zero_indices))[:num_to_zero]
            target_p[i, non_zero_indices[indices_to_zero]] = 0
    return target_p