import torch

def calculate_alpha(T, T_l, T_s):
    """
    Calculate gamma using the given formula.
    
    Args:
        T: Input temperature (scalar, np.array or torch.Tensor)
        T_m: Melting temperature parameter
        T_l: Liquidus temperature parameter
        T_s: Solidus temperature parameter
    
    Returns:
        gamma value with same type/shape as input T
    """
    T_m = 0.5*(T_l+T_s)
    numerator = 4 * (T - T_m)
    denominator = T_l - T_s
    erf_arg = numerator / denominator
    gamma = 0.5 * torch.erf(erf_arg) + 0.5
    return gamma
