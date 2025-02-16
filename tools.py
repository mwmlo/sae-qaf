import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from SpArX.sparx import FFNN, KMeansClusterer, GlobalMerger, LocalMerger
import sparse_autoencoder
import blobfile as bf
import torch


def extract_mlp(model, layer_index: int):
    """
    Extract the nth transformer model layer's MLP as a separate MLP.
    """
    W_in = model.W_in[layer_index].detach().numpy()
    b_in = model.b_in[layer_index].detach().numpy()
    W_out = model.W_out[layer_index].detach().numpy()
    b_out = model.b_out[layer_index].detach().numpy()

    nx, nf = W_in.shape[0], W_in.shape[1]

    intermediate_layer = Dense(units=nf, activation="gelu")
    intermediate_layer.build((None, nx))
    intermediate_layer.set_weights([W_in, b_in])

    output_layer = Dense(units=nx)
    output_layer.build((nx, nf))
    output_layer.set_weights([W_out, b_out])

    mlp = Sequential([
        intermediate_layer,
        output_layer
    ])
    mlp.build((None, nx))

    # Freeze the layers
    for layer in mlp.layers:
        layer.trainable = False

    mlp.summary()
    return mlp


def shrink_model_global(original_model, shrink_percentage):
  cluster_labels = KMeansClusterer.cluster(original_model, shrink_percentage)
  merged_model = GlobalMerger.merge(original_model, cluster_labels)
  return merged_model, cluster_labels


def shrink_model_local(original_model, shrink_percentage):
  cluster_labels = KMeansClusterer.cluster(original_model, shrink_percentage)
  merged_model = LocalMerger.merge(original_model, cluster_labels)
  return merged_model, cluster_labels


def get_original_mlp_for_sparx(mlp, ds_acts):
    shape = (768, 768*4, 768)
    weights = [layer.get_weights()[0] for layer in mlp.layers]
    bias = [layer.get_weights()[1] for layer in mlp.layers]
    activations = ["gelu", "linear"]

    restored_model = FFNN(shape, weights, bias, activations)
    restored_model.forward_pass(ds_acts)
    return restored_model


def output_infidelity(x, y):
    return np.linalg.norm(x - y)/len(x)

def output_mse(x, y):
  return ((x - y)**2).mean()

def output_r2(x, y):
  return 1 - (((x-y)**2).mean() / np.var(y))


def get_sparse_autoencoder(location, layer_index):
    with bf.BlobFile(sparse_autoencoder.paths.v4(location, layer_index), mode="rb") as f:
        state_dict = torch.load(f)
        autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
        return autoencoder
