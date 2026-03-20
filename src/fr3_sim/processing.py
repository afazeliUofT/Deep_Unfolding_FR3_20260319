from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class MmseOutput:
    v: tf.Tensor
    mse: tf.Tensor
    sinr: tf.Tensor


def mmse_combiners_and_mse(H: tf.Tensor, w: tf.Tensor, noise_var_watt: tf.Tensor | float) -> MmseOutput:
    """Return linear MMSE combiners, MSEs, and post-combining SINRs.

    Parameters
    ----------
    H : [S, B, U, Nr, M]
        Channel from each BS to each user.
    w : [S, B, M, K]
        Downlink beamformers for K = users-per-BS streams.
    """
    complex_dtype = H.dtype
    real_dtype = tf.float32 if complex_dtype == tf.complex64 else tf.float64

    S = int(H.shape[0])
    U = int(H.shape[2])
    Nr = int(H.shape[3])
    K = int(w.shape[3])

    g = tf.einsum("sburm,sbmk->sburk", H, w, optimize=True)
    Ryy = tf.einsum("sburk,sbunk->surn", g, tf.math.conj(g), optimize=True)
    eye_nr = tf.eye(Nr, batch_shape=[S, U], dtype=complex_dtype)
    Ryy = Ryy + tf.cast(noise_var_watt, complex_dtype) * eye_nr

    desired_parts = []
    eye_k = tf.eye(K, dtype=complex_dtype)
    B = int(H.shape[1])
    for b in range(B):
        block = g[:, b, b * K : (b + 1) * K, :, :]
        desired_parts.append(tf.einsum("surk,uk->sur", block, eye_k, optimize=True))
    h_des = tf.concat(desired_parts, axis=1)

    v = tf.linalg.solve(Ryy, h_des[..., tf.newaxis])[..., 0]
    gain = tf.reduce_sum(tf.math.conj(h_des) * v, axis=-1)
    mse = tf.maximum(tf.math.real(1.0 - gain), tf.cast(1e-9, real_dtype))
    sinr = tf.maximum(1.0 / mse - 1.0, tf.cast(0.0, real_dtype))
    return MmseOutput(v=v, mse=mse, sinr=sinr)
