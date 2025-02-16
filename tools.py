import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from SpArX.sparx import FFNN, KMeansClusterer, GlobalMerger, LocalMerger

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer

from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE

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


def shrink_model(original_model, shrink_percentage):
  cluster_labels = KMeansClusterer.cluster(original_model, shrink_percentage)
  merged_model = GlobalMerger.merge(original_model, cluster_labels)
  return merged_model, cluster_labels


def sparsify_mlp(mlp, ds_acts, shrink_pcs = [0.1, 0.3, 0.5, 0.7, 0.9]):
    shape = (768, 768*4, 768)
    weights = [layer.get_weights()[0] for layer in mlp.layers]
    bias = [layer.get_weights()[1] for layer in mlp.layers]
    activations = ["gelu", "linear"]

    restored_model = FFNN(shape, weights, bias, activations)
    restored_model.forward_pass(ds_acts)

    merged_models = []
    models_cluster_labels = []
    for pc in shrink_pcs:
        model, labels = shrink_model(restored_model, pc)
        model.model.summary()
        merged_models.append(model)
        models_cluster_labels.append(labels)

    return merged_models, models_cluster_labels