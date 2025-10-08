import torch
import torch.nn as nn
import torch.nn.functional as F

# Static Attention
def static_genetic_attention(parent1_feature_map, parent2_feature_map, attention_weights=[0.7, 0.3]):
    """
    Applies an attention mechanism to combine the feature maps from two parent networks into a child's feature map.
    
    Parameters:
    parent1_feature_map (torch.Tensor): Feature map tensor from Parent 1 (e.g., [batch_size, channels, height, width]).
    parent2_feature_map (torch.Tensor): Feature map tensor from Parent 2 (e.g., [batch_size, channels, height, width]).
    (Parent 2 is the initial child network)
    attention_weights (list or torch.Tensor): Weights representing the importance of each parent's feature map.
                                              Should have two values: [weight_for_parent1, weight_for_parent2].
    
    Returns:
    torch.Tensor: The resulting child feature map.
    """
    
    # Ensure both feature maps have the same shape
    if parent1_feature_map.shape != parent2_feature_map.shape:
        raise ValueError("Feature maps from both parents must have the same shape.")
    
    # Convert attention_weights to a tensor if necessary and normalize them
    attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
    attention_weights = attention_weights / torch.sum(attention_weights)  # Normalize to sum to 1

    # Apply the attention mechanism to the feature maps (element-wise weighted sum)
    child_feature_map = (attention_weights[0] * parent1_feature_map) + (attention_weights[1] * parent2_feature_map)
    
    return child_feature_map


# Dynamic attention
def dynamic_genetic_attention(query, keys, values):
    """
    Computes scaled dot-product attention.

    Parameters:
    query (torch.Tensor): Query tensor of shape (batch_size, channels, height, width).
    keys (torch.Tensor): Keys tensor of shape (num_parents, batch_size, channels, height, width).
    values (torch.Tensor): Values tensor of shape (num_parents, batch_size, channels, height, width).

    Returns:
    torch.Tensor: The output tensor (child feature map) after applying attention.
    """
    batch_size, channels, height, width = query.shape

    # Flatten the spatial dimensions (height and width)
    query_flattened = query.view(batch_size, channels, -1)  # Shape: [batch_size, channels, height * width]
    keys_flattened = keys.view(batch_size, channels, -1)    # Shape: [batch_size, channels, height * width]

    # Compute the dot product between query and keys along the channels dimension
    # This gives attention scores for each spatial position
    attention_scores = torch.einsum('bci,bci->bi', query_flattened, keys_flattened) / torch.sqrt(torch.tensor(channels, dtype=torch.float32))

    # Apply softmax to get the attention weights (across spatial locations, i.e., height * width)
    attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: [batch_size, height * width]

    # Flatten the values tensor for each spatial position
    values_flattened = values.view(batch_size, channels, -1)  # Shape: [batch_size, channels, height * width]

    # Weighted sum of values based on the attention weights
    child_flattened = torch.einsum('bi,bci->bci', attention_weights, values_flattened)

    # Reshape back to the original spatial dimensions
    child_feature_map = child_flattened.view(batch_size, channels, height, width)

    return child_feature_map

def dynamic_genetic_attention_2d(query, keys, values):
    B, C = query.shape
    scores = (query * keys).sum(dim=1, keepdim=True) / (C ** 0.5)  # [B, 1]

    # Apply softmax over the batch dimension (optional but unnecessary with [B, 1])
    attn_weights = torch.sigmoid(scores)  # [B, 1] — sigmoid for scalar gate

    # Weighted sum between original value and query (residual-like)
    output = attn_weights * values + (1 - attn_weights) * query  # [B, C]

    return output

def parent_child_attention(query_feature_map, parent_feature_maps):
    """
    Implements parent-child attention mechanism where the child feature map (query) is derived 
    from parent feature maps (keys and values).

    Parameters:
    query_feature_map (torch.Tensor): Query tensor of the child network feature map.
                                      Shape: (batch_size, channels, height, width)
    parent_feature_maps (torch.Tensor): Feature maps from parent networks. 
                                        Shape: (num_parents, batch_size, channels, height, width)

    Returns:
    torch.Tensor: Child feature map after attention.
    """
    # Scaling factor
    vlambda = 0.6
    # Keys and Values are both parent feature maps (could be different in a more complex model)
    keys = parent_feature_maps
    #print(f'========{keys.shape}')
    values = parent_feature_maps
    
    # ensemble = keys + query_feature_map
    
    # Apply scaled dot-product attention - First compute the memory content from parent (Memory ARN)
    child_feature_map = query_feature_map + vlambda*(values-dynamic_genetic_attention_2d(query_feature_map, keys, values))
    
    return child_feature_map



