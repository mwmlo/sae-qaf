import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import requests


def get_latent_features(base_activation, sae):
  # Focus on the endoftext token
  latents, _ = sae.encode(base_activation)

  # Get list of active latent features and order by latent scores
  latent_idxs = torch.nonzero(latents).flatten()
  latent_acts = latents[latent_idxs]

  acts_idxs = sorted(zip(latent_acts, latent_idxs), key=lambda x: x[0], reverse=True)
  return acts_idxs


def _retrieve_neuronpedia(layer_index, feature_index):
  url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer_index}-mlp-oai/{feature_index}"
  headers = {
    'Accept': 'application/json',
    'x-api-key' : "sk-np-Scp9TkTw71M2JcbNeVXDxMF3081GBnf8YY0wtKgAkf00"
  }
  try:
    response = requests.get(url, headers=headers)
  except:
    return [], []
  
  return response.json()


def get_explanations(feature_index, layer_index):
  response = _retrieve_neuronpedia(feature_index, layer_index)
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


def get_top_activating_prompts_for_latent_feature(layer_index, feature_index: int) -> Tuple[List[str], List[str]]:
  response = _retrieve_neuronpedia(layer_index, feature_index)

  descriptions = []
  explanations = response.get('explanations')
  for e in explanations:
    descriptions.append(e['description'])

  # Get top five activating prompts: context window +-3 tokens around top activating token
  prompts = []
  top_samples = response.get('activations')[:5]
  for sample in top_samples:
    max_value_token_index = sample['maxValueTokenIndex']
    prompt_tokens = sample['tokens'][max_value_token_index-3:max_value_token_index+4]
    prompts.append(''.join(prompt_tokens))

  return (descriptions, prompts)


def get_top_clusters_for_prompts(model, prev_layer_loc, prompts: List[str], sparx_model) -> List[List[int]]:
  concept_logits, concept_cache = model.run_with_cache(prompts)
  concept_acts = concept_cache[prev_layer_loc][:,-1].numpy().reshape(-1, 768)

  sparx_model.forward_pass(concept_acts)
  # Get cluster-neuron activations in MLP hidden layer
  prompt_cluster_acts = sparx_model.forward_pass_data[0]

  top_activating_clusters_per_prompt = [list(np.where(prompt_acts > 1.0)[0]) for prompt_acts in prompt_cluster_acts]
  return top_activating_clusters_per_prompt


def measure_similarity_in_activated_clusters(top_clusters: List[List[int]]) -> List[List[float]]:
  n_prompts = len(top_clusters)
  prompt_cluster_sim = [[0.0] * n_prompts for _ in range(n_prompts)]

  for i in range(n_prompts):
    for j in range(i, n_prompts):
      clusters_a = top_clusters[i]
      clusters_b = top_clusters[j]
      print(clusters_a, clusters_b)
      sim = calculate_similarity(clusters_a, clusters_b)
      prompt_cluster_sim[i][j] = sim
      prompt_cluster_sim[j][i] = sim

  return prompt_cluster_sim


def list_explanations_for_single_activation(activation, autoencoder):
  with torch.no_grad():
    exps = get_latent_features(activation, autoencoder)

  for act, idx in exps[:5]:
    descriptions, max_tokens = get_explanations(idx)
    print(f"Latent feature {idx}: {descriptions}; {max_tokens}")