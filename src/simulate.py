"""Simulación vectorizada de una política de revisión periódica (R, S) con ventas perdidas."""
import numpy as np


def simular_rs(demanda, S, revision, L, inv_inicial):
    """
    demanda:     (T, N) demanda diaria por producto
    S:           (T, N) nivel objetivo; solo se usa en los días de revisión
    revision:    (T,) booleano, True en los días en que se hace pedido
    L:           lead time en días (el pedido del día t llega al inicio del día t+L)
    inv_inicial: (N,) inventario al inicio del día 0

    Orden de cada día: llegan pedidos → se vende → si es día de revisión, se pide S − posición.
    """
    T, N = demanda.shape
    pipe = np.zeros((T + L + 1, N))          # pipe[k] = unidades que llegan al inicio del día k
    on_hand = inv_inicial.astype(float).copy()
    vendido = np.zeros((T, N))
    perdido = np.zeros((T, N))
    inventario = np.zeros((T, N))
    pedidos = np.zeros((T, N))

    for t in range(T):
        on_hand += pipe[t]
        v = np.minimum(on_hand, demanda[t])
        vendido[t] = v
        perdido[t] = demanda[t] - v
        on_hand -= v
        if revision[t]:
            transito = pipe[t + 1: t + L + 1].sum(axis=0)
            q = np.maximum(0.0, np.ceil(S[t] - on_hand - transito))
            pipe[t + L] += q
            pedidos[t] = q
        inventario[t] = on_hand

    return {"vendido": vendido, "perdido": perdido, "inventario": inventario, "pedidos": pedidos}