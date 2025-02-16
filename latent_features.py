import requests
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_latent_features(base_activation, sae):
  # Focus on the endoftext token
  latents, _ = sae.encode(base_activation)

  # Get list of active latent features and order by latent scores
  latent_idxs = torch.nonzero(latents).flatten()
  latent_acts = latents[latent_idxs]

  acts_idxs = sorted(zip(latent_acts, latent_idxs), key=lambda x: x[0], reverse=True)
  return acts_idxs


def get_explanations(feature_index, layer_index):
  url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer_index}-mlp-oai/{feature_index}"
  headers = {
    'Accept': 'application/json',
    'x-api-key' : "sk-np-Scp9TkTw71M2JcbNeVXDxMF3081GBnf8YY0wtKgAkf00"
  }
  try:
    response = requests.get(url, headers=headers)
  except:
    return [], []

  response = response.json()
  descriptions = []
  explanations = response.get('explanations')
  for e in explanations:
    descriptions.append(e['description'])

  # Get top three activating samples
  tokens = []
  top_samples = response.get('activations')[:5]
  for sample in top_samples:
    max_value_token_index = sample['maxValueTokenIndex']
    tokens.append(sample['tokens'][max_value_token_index])

  return descriptions, tokens


def activated_cluster_neurons(cluster_labels, cluster_idx, activation_cache, transformer_lens_loc):
  local_activation = activation_cache[transformer_lens_loc][0,-1].detach().clone()
  mask_neuron_idxs = list(np.where(cluster_labels != cluster_idx)[0])
  local_activation[mask_neuron_idxs] = 0.0
  return local_activation


# Get map of cluster index -> list of top 5 latent feature indices
def get_cluster_latents(num_clusters, autoencoder, cluster_labels, activation_cache):
  cluster_latents = {}

  for n in range(num_clusters):
    acts = activated_cluster_neurons(cluster_labels[0], n, activation_cache)
    acts_idxs = get_latent_features(acts, autoencoder)[:5]
    # Collect indices of top five latent features
    idxs = [act_idx[1].item() for act_idx in acts_idxs]
    cluster_latents[n] = idxs

  return cluster_latents


def calculate_similarity(latents_a, latents_b):
    # Convert lists to sets to find unique elements
    set_a = set(latents_a)
    set_b = set(latents_b)

    # Jaccard index: intersection over union
    shared_elements = set_a.intersection(set_b)
    total = set_a.union(set_b)
    similarity = len(shared_elements) / len(total)

    return similarity


def get_cluster_similarity_matrix(num_clusters, cluster_latents):
  cluster_sim = np.zeros((num_clusters, num_clusters))
  for i in range(num_clusters):
    for j in range(i, num_clusters):
      similarity_score = calculate_similarity(cluster_latents[i], cluster_latents[j])
      cluster_sim[i][j] = similarity_score
      cluster_sim[j][i] = similarity_score
  return cluster_sim


def plot_cluster_similarity_matrix(sim_matrix):
  # Plotting the heatmap using matplotlib
  plt.figure(figsize=(8, 6))
  plt.imshow(sim_matrix, cmap='coolwarm', interpolation='nearest')
  plt.colorbar(label='Similarity Score')

  plt.title('Similarity Scores Heatmap')
  plt.xlabel('Clusters')
  plt.ylabel('Clusters')
  plt.show()


def cluster_activations_for_prompt(model, prev_layer_loc, concept_tokens, sparx_mlp):
  concept_logits, concept_cache = model.run_with_cache(concept_tokens)
  concept_acts = concept_cache[prev_layer_loc][:,-1].numpy().reshape(-1, 768)

  sparx_mlp.forward_pass(concept_acts)
  # Get cluster-neuron activations in MLP hidden layer
  hidden_cluster_acts = sparx_mlp.forward_pass_data[0]

  plt.figure(figsize=(15, 2))
  plt.imshow(hidden_cluster_acts, cmap='coolwarm', aspect='auto')
  plt.colorbar(label='Activations')
  plt.show()

  top_activating_clusters = [list(np.where(acts > 1.0)[0]) for acts in hidden_cluster_acts]
  return top_activating_clusters