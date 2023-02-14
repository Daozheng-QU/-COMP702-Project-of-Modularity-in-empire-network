#Student Name: Daozheng QU
#Student ID:201518453
#University of Liverpool
#MSc of Data Science and AI
#COMP702 Project of Modularity in empire network
#Supervisor:Michele Zito,Keith Dures.

import networkx as nx
import numpy as np
import copy,sys
import matplotlib.pyplot as plt
import os


#function for single simulation given vertices number and iteration number, starting from an empty graph with no edges
def run_simulation(vertices_num, iter_num):
    # create empty graph with n vertices and no edges
    G = nx.empty_graph(vertices_num)
    # store history edge numbers in simulation
    edge_nums = list()
	# print('iteration {}'.format(k), end='\r')
    for k in range(iter_num):
        
    #Put the new edges into the edge_nums
        edge_nums.append(G.number_of_edges())

        # choose position (i,j) uniformly,generate a uniform random sample from vertices_num of size 2 without replacement.
        i, j = np.random.choice(vertices_num, 2, replace=False)
        # if contains the edge e, remove the edge, prevent duplication
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

#Establish a community division on the empire map and calculate the class of the maximum Q value
#which is convenient for quickly using the fast Newman algorithm
class Empire_Graph(object):


	def __init__(self, G):
		# Temperal empire graph for fast newman algorithm, which will be changed during the calculation
		self.Temp_G = G

		# Set the nodes of the temporary graph
		self.n = len(G.nodes())
		# Set the edges of the temporary graph
		self.m = len(G.edges())
		# create zeros adjacency matrix
		##nodes([[ 0.,  0.],
        #        [ 0.,  0.]])
		self.Adj_mat = np.mat(np.zeros((self.n, self.n)))
		# node index in adjacency matrix
		self.Node_Index = {}

		# Id of node i
		idxi = 0
		# Id of node j
		idxj = 0
		# Iterate each edge to create adjacency matrix
		for edge in self.Temp_G.edges():
			# Node i,j represents the edge
			node_i = edge[0]
			node_j = edge[1]
			# If node_i is not in the (self.Node_Index,empty dictionary), Id of storage node i
			if node_i not in self.Node_Index:
				self.Node_Index[node_i] = idxi
				idxi = idxi + 1
			# If node_j is not in the (self.Node_Index,empty dictionary), Id of storage node j
			if node_j not in self.Node_Index:
				self.Node_Index[node_j] = idxj
				idxj = idxj + 1

			# find node i and node j index in adjacency matrix
			node_i_index = self.Node_Index[node_i]
			node_j_index = self.Node_Index[node_j]
			# set adjacency matrix equal to one(unweights) at [node_i_index, node_j_index]
			self.Adj_mat[node_i_index, node_j_index] = 1
			self.Adj_mat[node_j_index, node_i_index] = 1
		# The community number of each node
		self.Node_group_Num = {}
		# The divisions for temporary graph
		self.division = {}
		# The node of the current iteration, The node group of the item at the current iteration
		for i, n in enumerate(G.nodes()):
			# Each node is divided into a community and a separate community group
			self.division[i] = [n]
			self.Node_group_Num[n] = i


# function for calculating delta Q
def calculate_delta_Q(empire_graph, division_i, division_j):
	m = empire_graph.m
	# the fraction of edges that fall within communities
	a_i = 0
	a_j = 0
	# the sum of degrees for all nodes in division i
	k_i = 0
	# the sum of degrees for all nodes in division j
	k_j = 0
	# eij be the fraction of edges in the network that connect vertices in group i to those in group j.
	e_ij = 0

	# calculate the sum of degrees for all nodes in division i
	# calcualte original a_i(unchanged a_i of Newman formula) which is k_i
	for i, node_i in enumerate(division_i):
		node_i_index = empire_graph.Node_Index[node_i]
		k_i += empire_graph.Adj_mat[node_i_index].sum()

	# calculate the sum of degrees for all nodes in division j
	# calcualte original a_j(unchanged a_j of Newman formula) which is k_j
	for j, node_j in enumerate(division_j):
		node_j_index = empire_graph.Node_Index[node_j]
		k_j += empire_graph.Adj_mat[node_j_index].sum()

	# calcualte e_ij
	# calcualte original e_ij(unchanged e_ij of Newman formula)
	for i, node_i in enumerate(division_i):
		node_i_index = empire_graph.Node_Index[node_i]
		for j, node_j in enumerate(division_j):
			node_j_index = empire_graph.Node_Index[node_j]
			e_ij += empire_graph.Adj_mat[node_i_index, node_j_index]
	# Divide the number of edges by 2 times according to the corrected formula by Newman himself
	a_i = k_i / float(2 * m)
	a_j = k_j / float(2 * m)
	e_ij = e_ij / float(2 * m)
	return 2 * (e_ij - a_i * a_j)


# function for calculating the value Q
def calculate_Q(empire_graph, division):
	m = empire_graph.m
	Q = 0

	# calcualte e_ii and a_i for each division
	for group in division:
		# eii represents the ratio of the number of edges within community i to the number of edges in the entire network
		e_ii = 0
		# the sum of degrees divide 2 times edge for all nodes in division i
		e_i = 0
		a_i = 0
		# the sum of degrees for all nodes in division i
		k_i = 0
		# the sum of degrees for all nodes in division j
		k_j = 0

		# calcualte a_i
		# calcualte original a_i(unchanged a_i of Newman formula) which is k_i
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
		# Divide the number of edges by 2 times according to the corrected formula by Newman himself
		a_i = k_i / float(2 * m)
		e_ii = e_i / float(2 * m)
		Q += (e_ii - a_i ** 2)

	return Q

#Define a fast Newman algorithm function on the empire graph
def Fast_Newman_Algo(empire_graph):
	# Retention of community division at the maximum Q value
	division_ret = None
	# maximum Q value
	max_q = float("-inf")
	#The loop continues to run when the community division is not merged into only one community.
	while len(empire_graph.division) > 1:
		#Temporary largest DeltaQ
		Temp_Max_Delta_Q = float("-inf")
		#The edges to be merged according to Delta Q
		Merge_edge = None
		#All Delta Q
		All_Delta_Q = []

		# Iterator all edges in temperal graph, repeatedly join communities together in pairs,[0,1]represent edge.
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

			# Find maximal Delta Q
			if cur_det_Q > Temp_Max_Delta_Q:
				Temp_Max_Delta_Q = cur_det_Q
				#The edge to be merged
				Merge_edge = edge
		#Stop if there is no merged edge
		if Merge_edge is None:
			break

		# join communities together in pairs with maximal delta Q
		Indexi = empire_graph.Node_group_Num[Merge_edge[0]]
		Indexj = empire_graph.Node_group_Num[Merge_edge[1]]
		empire_graph.division[Indexi].extend(empire_graph.division[Indexj])
		#Merge community and delete duplicate community
		for node in empire_graph.division[Indexj]:
			empire_graph.Node_group_Num[node] = Indexi
		del empire_graph.division[Indexj]
		# remove the edge after joining
		empire_graph.Temp_G.remove_edge(Merge_edge[0], Merge_edge[1])

		# Find the divisions with maximal Q
		groups = copy.deepcopy(list(empire_graph.division.values()))
		cur_Q = calculate_Q(empire_graph, groups)
		#If the current Q is greater than the maximum Q, it becomes the new maximum Q
		if cur_Q > max_q:
			max_q = cur_Q
		#Results retention division of the community in case the maximum value of Q
			division_ret = groups

	return division_ret, max_q
#The interval of the number of nodes in a random planar graph, and the interval of the number of nodes
for vertice_num in range(500, 5000, 500):
	#Number of nodes in the planar graph
	vertices_num = vertice_num

	#The number of iterations
	iter_num = 10000
	# Number of nodes in a single random planar graph running experiments
	epoch = 20

	# empire graph, r value
	r_values = [5]
	#Create a save path for the result folder
	save_path = "./results/"
	#If it does not exist, create a path to save the result folder
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	#Create a save path according to the interval of the number of nodes in the random planar graph to save the result folder
	save_path = "./results/%d" % (vertices_num)
	#If the result folder of the save path already exists, delete the folder information such as the save path
	if os.path.exists(save_path):
		for filename in os.listdir(save_path):
			os.remove(save_path + "/" + filename)
	#If it does not exist, create a path to save the result folder
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	#Open that write the output result information into the txt file, if there is the original data will be overwritten.
	f = open(save_path + "/result.txt", "w")
##Store max_Q value according to r value
	all_Q = {}
	#Run the experiment at the specified number of times
	for i in range(epoch):
		# random planar graph
		graph_ori, edge_nums = run_simulation(vertices_num, iter_num)
		#Set the number of iterations in the range[,)of iter_num.
		iterations = np.arange(iter_num)
		#run random planar graphs and simulation graphs under the number of epoch
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
			# empire graph
			graph_r1 = create_empire(graph_ori, r)
			#Remove the self-loop of the empire graph node
			graph_r1.remove_edges_from(nx.selfloop_edges(graph_r1))
			#Use the empire graph be temporary graph
			G = create_empire(graph_ori, r)
			#Perform fast newman algorithm on the empire graph
			empire_graph = Empire_Graph(G)
			#Calculate the result of community division and retention under the maximum Q value
			division, max_Q = Fast_Newman_Algo(empire_graph)
			#Store max_Q value according to r value, if r's max Q in the  all Q add max Q
			if r in all_Q:
				all_Q[r].append(max_Q)
			#else r's max Q= max Q
			else:
				all_Q[r] = [max_Q]
			#print iteration order  of each r value,max Q,division.
			result = "Iteration %d: r = %d, the maximal Q is %.3f, division: %s" % (i, r, max_Q, division)
			f.write(result + "\n")
			print(result)

			#print empire gragph
			plt.figure(figsize=(20, 20))
			plt.title('generated empire graph with {} vertices'.format(vertices_num//r))
			nx.draw(graph_r1, node_size=50, alpha=0.8)
			plt.savefig('%s/empire_graph_%d_r%d.jpg' % (save_path, i, r) )
			# plt.show()

	for r in all_Q:
		# get average max Q
		max_Q = sum(all_Q[r]) / float(len(all_Q[r]))
		#The average maximum Q value under each r value
		result = "r = %d: the average Q is %.3f" % (r, max_Q)
		f.write(result + "\n")
		print(result)
	#close the file f
	f.close()