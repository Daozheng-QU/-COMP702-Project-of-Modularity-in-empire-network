#Student Name: Daozheng QU
#Student ID:201518453
#University of Liverpool
#MSc of Data Science and AI
#COMP702 Project of Modularity in empire network
#Supervisor:Michele Zito,Keith Dures.

import networkx as nx
import numpy as np
import pandas as pd
import copy, sys
import matplotlib.pyplot as plt
import os


#function for single simulation given vertices number and iteration number, starting from an empty graph with no edges
def run_simulation(vertices_num, iter_num):
    # create empty graph with n vertices and no edges
    G = nx.empty_graph(vertices_num)
    # store history edge numbers in simulation
    edge_nums = list()

    for k in range(iter_num):
        # print('iteration {}'.format(k), end='\r')
        #Put the new edges into the edge_nums
        edge_nums.append(G.number_of_edges())

        # choose position (i,j) uniformly,generate a uniform random sample from vertices_num of size 2 without replacement.
        i, j = np.random.choice(vertices_num, 2, replace=False)
        # if contains the edge e, remove the edge
        if G.has_edge(i, j):
            G.remove_edge(i, j)
        # if does not contains the edge, add edge (i,j)
        else:
            G_next = G.copy()
            G_next.add_edge(i, j)
            # if still planarity, leave the newly added edge
            if nx.check_planarity(G_next)[0]:
                G = G_next.copy()
            # else, the next state remains the same as the previous one
            else:
                pass
    return G, edge_nums

#Create a function to generate an empire graph
def create_empire(G, r):
    #Random graph edges, request edge attributes from the list()
    G_edges = list(G.edges)
    #random graph nodes,request node attributes from the list()
    G_vnum = len(list(G.nodes))
    # nodes of the empire graph
    EG_vnum = int(G_vnum/r)
    #store history edge numbers
    EG_edges_list = []
    for e in G_edges:
        #Put the new edges into the eg_edges_list
        EG_edges_list.append((int((e[0])/r), int((e[1])/r)))
    #Create an empty graph with non vertices and edges
    EG = nx.Graph()
    #Add nodes of the empire graph
    EG.add_nodes_from(range(EG_vnum))
    #Add edges of the empire graph
    EG.add_edges_from(EG_edges_list)
    return EG




class Empire_Graph(object):
	"""docstring for Empire_Graph"""
	def __init__(self, G):
		# Temperal empire graph for fast newman algorithm, which will be changed during the calculation
		self.Temp_G = G

		# create zeros adjacency matrix
		self.n = len(G.nodes()) 
		self.m = len(G.edges())
		self.Adj_mat = np.mat(np.zeros((self.n, self.n)))
		# node index in adjacency matrix
		self.Node_Index = {}
		# Iterate each edge to create adjacency matrix
		idx = 0
		for edge in self.Temp_G.edges():
			node_i = edge[0]
			node_j = edge[1]
			if node_i not in self.Node_Index:
				self.Node_Index[node_i] = idx
				idx  = idx + 1
			if node_j not in self.Node_Index:
				self.Node_Index[node_j] = idx
				idx  = idx + 1

			# find node i and node j index in adjacency matrix
			node_i_index = self.Node_Index[node_i]
			node_j_index = self.Node_Index[node_j]
			# set adjacency matrix equal to one at [node_i_index, node_j_index]
			self.Adj_mat[node_i_index, node_j_index] = 1
			self.Adj_mat[node_j_index, node_i_index] = 1
		# The community number of each node
		self.Node_group_Num = {}
		# The partitions for empire graph
		self.division = {}
		for i, n in enumerate(G.nodes()):
			#every node will be a division
			self.division[i] = [n]
			self.Node_group_Num[n] = i

# function for calculating delta Q
def calculate_delta_Q(empire_graph, division_i, division_j):
	m = empire_graph.m
	a_i = 0
	a_j = 0
	#the sum of degrees for all nodes in partition i
	k_i = 0
	#the sum of degrees for all nodes in partition j
	k_j = 0
	e_ij = 0

	# calculate the sum of degrees for all nodes in partition i
	# calcualte a_i
	for i, node_i in enumerate(division_i):
		node_i_index = empire_graph.Node_Index[node_i]
		k_i += empire_graph.Adj_mat[node_i_index].sum()

	# calculate the sum of degrees for all nodes in partition j
	# calcualte a_j
	for j, node_j in enumerate(division_j):
		node_j_index = empire_graph.Node_Index[node_j]
		k_j += empire_graph.Adj_mat[node_j_index].sum()

	# calcualte e_ij
	for i, node_i in enumerate(division_i):
		node_i_index = empire_graph.Node_Index[node_i]
		for j, node_j in enumerate(division_j):
			node_j_index = empire_graph.Node_Index[node_j]
			e_ij += empire_graph.Adj_mat[node_i_index, node_j_index]

	a_i = k_i / float(2 * m)
	a_j = k_j / float(2 * m)
	e_ij = e_ij / float(2 * m)
	return 2 * (e_ij - a_i * a_j) 

# function for calculating the value Q
def calculate_Q(empire_graph, division):
	m = empire_graph.m
	Q = 0

	# calcualte e_ii and a_i for each devision
	for group in division:
		e_i = 0
		a_i = 0
		k_i = 0
		k_j = 0

		# calcualte a_i
		for i, node_i in enumerate(group):
			node_i_index = empire_graph.Node_Index[node_i]
			k_i += empire_graph.Adj_mat[node_i_index].sum()

		# calcualte the sum of e_ii
		for i, node_i in enumerate(group):
			node_i_index = empire_graph.Node_Index[node_i]
			for j, node_j in enumerate(group):
				node_j_index = empire_graph.Node_Index[node_j]
				e_i += empire_graph.Adj_mat[node_i_index, node_j_index]

		# Calculate Q
		a_i = k_i / float(2 * m)
		e_i = e_i / float(2 * m)
		Q += (e_i - a_i ** 2)

	return Q

def Fast_Newman_Algo(empire_graph):
	division_ret = None
	max_q = float("-inf")
	while len(empire_graph.division) > 1:
		Temp_Max_Delta_Q = float("-inf")
		Max_Merge = None
		All_Delta_Q = []

		# Iterator all edges in temperal graph, repeatedly join communities together in pairs,
		# choosing at each step the join that results in the greatest increase (or smallest decrease) in Q.
		for edge in empire_graph.Temp_G.edges():
			node_i = edge[0]
			node_j = edge[1]
			# Get communitiy number for node i and node j
			Indexi = empire_graph.Node_group_Num[node_i]
			Indexj = empire_graph.Node_group_Num[node_j]

			# Will not be considered, if node i and node j are in same communitiy
			if Indexi == Indexj:
				continue

			# calcualte delta Q before and after joining communitiy i and communitiy j
			cur_det_Q = calculate_delta_Q(empire_graph, empire_graph.division[Indexi], empire_graph.division[Indexj])
			All_Delta_Q.append([cur_det_Q, edge])

			# Find maximal Q
			if cur_det_Q > Temp_Max_Delta_Q:
				Temp_Max_Delta_Q = cur_det_Q
				Max_Merge = edge

		if Max_Merge is None:
			break

		# If multiple delta Q are equal, join communities together in pairs with maximal Q
		if len(All_Delta_Q) > 1:
			All_Delta_Q = sorted(All_Delta_Q, key = lambda x : x[0])
			tmp_max_Q = float("-inf")
			for each_det_Q in All_Delta_Q:
				if All_Delta_Q[-1][0] == each_det_Q[0]:
					tmp_Group = copy.deepcopy(empire_graph.division)
					tmp_edge = each_det_Q[1]
					Indexi = empire_graph.Node_group_Num[tmp_edge[0]]
					Indexj = empire_graph.Node_group_Num[tmp_edge[1]]
					tmp_Group[Indexi].extend(tmp_Group[Indexj])
					del tmp_Group[Indexj]
					units = copy.deepcopy(list(tmp_Group.values()))
					cur_Q = calculate_Q(empire_graph, units)
					if cur_Q > tmp_max_Q:
						tmp_max_Q = cur_Q
						Max_Merge = tmp_edge
				else:
					break
		
		# join communities together in pairs with maximal delta Q
		Indexi = empire_graph.Node_group_Num[Max_Merge[0]]
		Indexj = empire_graph.Node_group_Num[Max_Merge[1]]
		empire_graph.division[Indexi].extend(empire_graph.division[Indexj])
		for node in empire_graph.division[Indexj]:
			empire_graph.Node_group_Num[node] = Indexi
		del empire_graph.division[Indexj]
		# remove the edge after joining
		empire_graph.Temp_G.remove_edge(Max_Merge[0], Max_Merge[1])

		# Find the devisions with maximal Q
		units = copy.deepcopy(list(empire_graph.division.values()))
		cur_Q = calculate_Q(empire_graph, units)
		if cur_Q > max_q:
			max_q  = cur_Q
			division_ret = units

	return division_ret, max_q

for vertice_num in range(100, 1000, 100):
	#Number of nodes in the planar graph
	vertices_num = vertice_num

	#The number of iterations
	iter_num = 20000

	epoch = 20

	# empire graph, r
	r_values = [2, 4, 5]

	save_path = "./results/"
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	save_path = "./results/%d" % (vertices_num)
	if os.path.exists(save_path):
		for filename in os.listdir(save_path):
			os.remove(save_path + "/" + filename)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	f = open(save_path + "/result.txt", "w")

	all_Q = {}
	for i in range(epoch):
		# random planar graph
		graph_ori, edge_nums = run_simulation(vertices_num, iter_num)
		#Set the number of iterations in the range[,)of iter_num.
		iterations = np.arange(iter_num)

		is_show = True
		if is_show:
			#print simulation
			plt.figure()
			plt.title('a typical simulation with {} vertices and {} iterations'.format(vertices_num, iter_num))
			plt.plot(iterations, edge_nums)
			plt.savefig('%s/simulation_%d.jpg' % (save_path, i))
			# plt.show()

			#print random graph
			plt.figure(figsize=(20, 20))
			plt.title('generated random planar graph with {} vertices'.format(vertices_num))
			nx.draw_planar(graph_ori, node_size=50, alpha=0.8)
			plt.savefig('%s/planar_%d.jpg' % (save_path, i))
			# plt.show()
				
		for r in r_values:
			graph_r1 = create_empire(graph_ori, r)
			#Remove the self-loop of the empire graph node
			graph_r1.remove_edges_from(nx.selfloop_edges(graph_r1))
			
			G = create_empire(graph_ori, r)
			empire_graph = Empire_Graph(G)
			partition, max_Q = Fast_Newman_Algo(empire_graph)
			if r in all_Q:
				all_Q[r].append(max_Q)
			else:
				all_Q[r] = [max_Q]

			result = "Iteration %d: r = %d, the maximal Q is %.3f, division: %s" % (i, r, max_Q, partition)
			f.write(result + "\n")
			print(result)

			#print empire gragph
			plt.figure(figsize=(20, 20))
			plt.title('generated empire graph with {} vertices'.format(vertices_num//r))
			nx.draw(graph_r1, node_size=50, alpha=0.8)
			plt.savefig('%s/empire_graph_%d_r%d.jpg' % (save_path, i, r) )
			# plt.show()

	for r in all_Q:
		max_Q = sum(all_Q[r]) / float(len(all_Q[r]))
		result = "r = %d: the average Q is %.3f" % (r, max_Q)
		f.write(result + "\n")
		print(result)
	
	f.close()
