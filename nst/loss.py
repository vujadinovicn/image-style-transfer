import torch

def compute_content_cost(content_output, generated_output, content_layer):
    a_C = content_output[content_layer]
    a_G = generated_output[content_layer]

    m, n_C, n_H, n_W = a_C.size()
    a_C_unrolled = a_C.view(m, n_C, n_H * n_W)

    m, n_C, n_H, n_W = a_G.size()
    a_G_unrolled = a_G.view(m, n_C, n_H * n_W)

    return torch.sum((a_C_unrolled - a_G_unrolled)**2) / (4 * n_H * n_W * n_C)

def compute_style_cost_for_specific_layer(a_S, a_G):
    _, n_C, n_H, n_W = a_S.size()
    a_S = a_S.view(n_C, n_H*n_W)

    _, n_C, n_H, n_W = a_G.size()
    a_G = a_G.view(n_C, n_H*n_W)

    GS = torch.mm(a_S, a_S.t())
    GG = torch.mm(a_G, a_G.t())

    return 0.5 / (n_H * n_W * n_C)**2 * torch.sum((GS - GG)**2)

def compute_style_cost(style_image_output, generated_image_output, style_layers):
    J_style = 0

    a_S = style_image_output
    a_G = generated_image_output
    i = 0
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    for style_layer in style_layers[:-1]:
        J_style_layer = compute_style_cost_for_specific_layer(a_S[style_layer], a_G[style_layer])
        J_style += weights[i] * J_style_layer
        i+=1

    return J_style

def total_cost(J_content, J_style, alpha=5, beta=80):
    return alpha * J_content + beta * J_style