import keras
import keras_hub
import tensorflow as tf
from tensorflow.keras.models import Sequential

def get_gpt2_mlp(layer_index: int):
    # Use GPT2 because MLP layers are simple + pre-trained SAEs available
    model = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Access the nth transformer layer (skipping first 6 embedding layers)
    layer = model.layers[2].layers[6+layer_index]

    # Extract the feed-forward network (MLP): (768, 768*4) -> (768*4, 768)
    ff_intermediate = layer._feedforward_intermediate_dense
    ff_output = layer._feedforward_output_dense

    # Extract the 6th transformer layer's MLP as a separate MLP
    mlp_model = Sequential([ff_intermediate, ff_output])

    # Freeze the layers
    for layer in mlp_model.layers:
        layer.trainable = False

    mlp_model.summary()
    return mlp_model