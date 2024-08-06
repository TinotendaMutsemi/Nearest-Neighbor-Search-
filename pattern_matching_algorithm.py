import numpy as np

def pattern_matching_algorithm(features, agent_params, partitions, matching_algorithm):
    """
    Implements the Pattern-Matching Algorithm (PMA).

    Args:
    features: Current features xt.
    agent_params: List of tuples, each containing (k, l, s_n, tau) for each agent.
    partitions: List of data partitions {p`}.
    matching_algorithm: Function that implements the Matching Algorithm.

    Returns:
    Ht_plus_1: Updated agent-controls for all agents.
    """
    
    # Number of agents
    num_agents = len(agent_params)
    
    # Initialize agent-controls
    Ht_plus_1 = np.zeros((num_agents, features.shape[1]))

    # Generate agent-controls for each agent
    for n in range(num_agents):
        k, l, s_n, tau = agent_params[n]
        Ht_plus_1[n] = matching_algorithm(tau, partitions, features, k, l, s_n)

    return Ht_plus_1
