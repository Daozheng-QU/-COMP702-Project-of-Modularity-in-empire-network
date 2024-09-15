import networkx as nx  # Library to handle graph creation and manipulation
import numpy as np  # Numpy for efficient array handling and matrix operations
import matplotlib.pyplot as plt  # Matplotlib for graph plotting and visualization
import random  # Random module for selecting random edges during simulation
import math  # Math for various mathematical operations like ceiling (⌈ |E| / N ⌉)

def run_simulation(vertices_num, iter_num):
    """
    Generates a random planar graph by adding or removing edges while ensuring that the graph remains planar.

    Parameters:
    - vertices_num: Number of nodes (vertices) in the graph.
    - iter_num: Number of iterations to add or remove edges.
    
    Returns:
    - G: A planar graph generated after iter_num iterations.
    - edge_nums: List of edge counts at each iteration, useful for tracking edge changes.
    """
    G = nx.empty_graph(vertices_num)  # Start with an empty graph (only nodes, no edges)
    edge_nums = []  # List to store the number of edges at each step

    # Perform iter_num iterations of edge addition/removal
    for k in range(iter_num):
        edge_nums.append(G.number_of_edges())  # Track the current number of edges
        # Randomly choose two distinct nodes (i, j) to either add or remove an edge between them
        i, j = np.random.choice(vertices_num, 2, replace=False)
        if G.has_edge(i, j):
            G.remove_edge(i, j)  # If an edge exists, remove it
        else:
            G_next = G.copy()  # Make a copy of the current graph
            G_next.add_edge(i, j)  # Try to add an edge between i and j
            if nx.check_planarity(G_next)[0]:  # Check if the new graph is still planar
                G = G_next  # If it's still planar, update G with the new edge

    return G, edge_nums  # Return the final graph and edge count history

def compute_similarity_matrix(graph):
    """
    Computes a similarity matrix where each element represents the similarity between two nodes based on shared neighbors.

    Parameters:
    - graph: A NetworkX graph object representing the graph.
    
    Returns:
    - S: An n x n matrix (where n is the number of nodes) where S[i, j] represents the similarity between node i and node j.
    """
    n = len(graph.nodes())  # Get the number of nodes in the graph
    S = np.zeros((n, n))  # Initialize the similarity matrix with zeros

    # Compute the similarity based on the number of shared neighbors between each pair of nodes
    for i in graph.nodes():
        for j in graph.nodes():
            if j in graph.neighbors(i):  # Only consider nodes that are neighbors
                shared_neighbors = len(list(nx.common_neighbors(graph, i, j)))  # Count shared neighbors
                S[i][j] = shared_neighbors  # Set the similarity value

    # Normalize the matrix so that values are between 0 and 1
    max_sim = np.max(S)  # Get the maximum similarity value
    if max_sim > 0:
        S /= max_sim  # Normalize by dividing each entry by the maximum similarity
    return S  # Return the normalized similarity matrix

def markov_similarity_enhancement(S, max_iters):
    """
    Enhances the similarity matrix by applying a Markov process. This captures higher-order similarity by iterating matrix multiplications.

    Parameters:
    - S: The initial similarity matrix.
    - max_iters: The number of iterations for the Markov process (typically set to ⌈ |E| / N ⌉).
    
    Returns:
    - R: The enhanced similarity matrix after max_iters iterations.
    """
    R = S.copy()  # Start with the initial similarity matrix (R is updated iteratively)
    A = S.copy()  # Use the adjacency matrix (in this case, the similarity matrix is treated similarly)

    # Perform max_iters iterations of matrix multiplication to enhance the similarity matrix
    for _ in range(max_iters):
        R = np.dot(R, A)  # Multiply R by A to propagate similarity information
    return R  # Return the enhanced similarity matrix

def recurrence_relation(S, r, steps):
    """
    Applies the recurrence relation to update the transition probabilities. This simulates the evolution of the system over time.

    Parameters:
    - S: The similarity matrix representing transitions between nodes.
    - r: The initial transition matrix (usually an identity matrix).
    - steps: Number of steps for the recurrence relation (iterations).
    
    Returns:
    - r: The updated transition matrix after the specified number of steps.
    """
    for _ in range(steps):
        r = np.dot(r, S)  # Perform matrix multiplication to update the transition probabilities
    return r  # Return the updated transition matrix

def form_initial_communities(graph, similarity_matrix):
    """
    Forms initial communities by grouping nodes that are most similar based on the similarity matrix.

    Parameters:
    - graph: A NetworkX graph object representing the graph.
    - similarity_matrix: A matrix where each element represents the similarity between pairs of nodes.
    
    Returns:
    - communities: A list of sets where each set represents a community of nodes.
    """
    communities = []  # Initialize a list to store the communities
    node_list = list(graph.nodes())  # Get the list of nodes

    # For each node, find the most similar node and group them into the same community
    for node in node_list:
        most_similar_node = node_list[np.argmax(similarity_matrix[node])]  # Find the node with the highest similarity
        found = False
        for community in communities:  # Check if the node or its most similar node are already in a community
            if node in community or most_similar_node in community:
                community.update([node, most_similar_node])  # Add both nodes to the same community
                found = True
                break
        if not found:  # If neither node is in a community, create a new community
            communities.append({node, most_similar_node})
    return communities  # Return the list of initial communities

def merge_small_communities(communities, threshold):
    """
    Merges small communities if their size is below a specified threshold.

    Parameters:
    - communities: A list of sets representing communities of nodes.
    - threshold: The minimum size a community should have; smaller communities will be merged.
    
    Returns:
    - communities: The updated list of communities after merging small ones.
    """
    i = 0
    # Iterate over the communities to check for small communities and merge them
    while i < len(communities):
        if len(communities[i]) < threshold:  # Check if the community size is below the threshold
            best_merge_index = None
            best_size = float('inf')  # Start with an infinitely large size for comparison
            for j in range(len(communities)):
                if i != j and len(communities[j]) < best_size:  # Find the best community to merge with
                    best_size = len(communities[j])
                    best_merge_index = j
            if best_merge_index is not None:  # If a merge is found, update the communities
                communities[best_merge_index].update(communities[i])
                del communities[i]  # Remove the merged community from the list
                continue
        i += 1  # Move to the next community
    return communities  # Return the merged communities

def compute_modularity(graph, communities):
    """
    Computes the modularity of the graph's community structure. Modularity measures the strength of division into communities.

    Parameters:
    - graph: A NetworkX graph object representing the graph.
    - communities: A dictionary mapping each node to its community.
    
    Returns:
    - Q: The modularity score of the community structure.
    """
    m = graph.number_of_edges()  # Get the total number of edges in the graph
    A = nx.adjacency_matrix(graph).todense()  # Get the adjacency matrix as a dense matrix
    degrees = dict(graph.degree())  # Get the degree (number of connections) for each node
    Q = 0.0  # Initialize the modularity score

    # Iterate over all pairs of nodes to calculate the modularity score
    for i in graph.nodes():
        for j in graph.nodes():
            if communities[i] == communities[j]:  # Only consider pairs of nodes in the same community
                Q += A[i, j] - (degrees[i] * degrees[j]) / (2 * m)  # Modularity formula

    Q /= (2 * m)  # Normalize the modularity score
    return Q  # Return the modularity score

def main():
    vertices_num = 100  # Number of nodes in the graph
    iter_num = 1000  # Number of iterations to modify the graph's edges
    lambda_threshold = 3  # Minimum size for a community; smaller communities will be merged

    # Generate a random planar graph
    graph, edge_nums = run_simulation(vertices_num, iter_num)
    
    # Plot the graph
    plt.figure(figsize=(8, 6))  # Set the figure size
    nx.draw(graph, node_color='lightblue', with_labels=True, node_size=500)  # Draw the graph with labels
    plt.show()  # Display the plot

    # Step 1: Compute the similarity matrix (state transition matrix)
    S = compute_similarity_matrix(graph)

    # Step 2: Apply Markov similarity enhancement to capture higher-order similarities
    max_iters = math.ceil(graph.number_of_edges() / vertices_num)  # Maximum iterations set to ⌈ |E| / N ⌉
    enhanced_similarity = markov_similarity_enhancement(S, max_iters)  # Enhance the similarity matrix
    print(f"Enhanced similarity matrix after {max_iters} iterations:\n", enhanced_similarity)

    # Step 3: Form initial communities based on similarity
    initial_communities = form_initial_communities(graph, enhanced_similarity)
    print("Initial communities:", initial_communities)

    # Step 4: Merge small communities based on the size threshold
    final_communities = merge_small_communities(initial_communities, lambda_threshold)
    print("Final communities:", final_communities)

    # Step 5: Compute the modularity of the network's community structure
    communities_dict = {node: idx for idx, community in enumerate(final_communities) for node in community}
    modularity = compute_modularity(graph, communities_dict)
    print(f"Modularity of the network: {modularity}")

if __name__ == "__main__":
    main()  # Run the entire process from graph generation to modularity calculation
