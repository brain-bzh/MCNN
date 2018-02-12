import numpy as np
import itertools
import tqdm
from collections import deque
import random
import argparse
import functools
import pickle


global best_score, best_translation

def read_graph(file_name):
    f = open(file_name,"r")
    lines = f.readlines()
    for index,line in enumerate(lines):
        if index == 0:
            N = int(line)
            A = np.zeros((N,N))
        else:
            splitted = line.replace("\n","").split(" ")
            for value in splitted:
                value = int(value)
                A[index-1,value] = 1
    return A

def generate_grid_graph(W=4,H=4):
    N = W*H
    A = np.zeros((N,N))
    for i in range(N):
        if i % H > 0:
            A[i,i-1] = 1
        if (i+1) % H > 0:
            A[i,i+1] = 1
        if i+H < N:
            A[i,i+H] = 1
        if i-H >= 0:
            A[i,i-H] = 1
    return A

def get_center(A):
    min_distance = np.inf
    center_node = -1
    for node in tqdm.tqdm(range(A.shape[0])):
        distance_dict = get_distance(A,node)
        distance = 0
        for v in distance_dict.values():
            distance += v
        if distance < min_distance:
            center_node = node
            min_distance = distance 
    return min_distance, center_node

def get_distance(A, start):
    visited, stack = dict(), [(start,0)]
    while stack:
        vertex, distance = stack.pop()
        if vertex not in visited or visited[vertex] > distance:
            visited[vertex] = distance
            neighbours = np.where(A[vertex] > 0)[0]
            neighbours_to_visit = list()
            for a in neighbours:
                neighbours_to_visit.append((a,distance+1))
            stack.extend(neighbours_to_visit)
    return visited

def compute_score(arguments,initial_graph,graph,index,translation,l,gts,stg):
    def loss_score():
        score = 0
        for i in range(index+1):
            if translation[i] == -1:
                score += 1
        return score
    def injection_score():
        score = 0
        for i in range(index+1):
            for j in range(i):
                if translation[i] == translation[j] and translation[i] != -1:
                    score += 1
        return score
    def edge_constrained_score():
        score = 0
        for i in range(index+1):
            if translation[i] != -1 and translation[i] not in np.where(graph[stg[i]] > 0)[0]:
                score += 1
        return score
    def snp_score():
        score = 0
        for i in range(index+1):
            if translation[i] != (-1):
                for j in range(index+1):
                    if translation[j] != (-1):
                        try:
                            if stg[i] in np.where(initial_graph[stg[j]] > 0)[0] and not (translation[i] in np.where(initial_graph[translation[j]] > 0)[0]):
                                score += 1
                            if stg[i] not in np.where(initial_graph[stg[j]] > 0)[0] and translation[i] in np.where(initial_graph[translation[j]] > 0)[0]:
                                score += 1
                        except Exception as e:
                            print(e)
        return score
    score = arguments.alpha * loss_score() + arguments.beta*injection_score() + arguments.gamma*edge_constrained_score() + arguments.delta*snp_score() 
    score += 1000000 if translation[0] not in np.where(graph[stg[0]] > 0)[0] else 0
    return score

def search(arguments,initial_graph,graph,l,gts,stg):
    global best_score, best_translation
    best_score = np.inf
    best_translation = []
    def in_search(score,index,translation):
        global best_score, best_translation
        if index == l:
            if score < best_score:
                best_score = score
                best_translation = translation.copy()
        else:
            for image in range(-1, len(stg)):
                translation[index] = stg[image] if image > -1 else -1
                new_score = compute_score(arguments,initial_graph,graph,index,translation,l,gts,stg)
                if new_score < best_score:
                    in_search(new_score,(index + 1),translation)         
    in_search(0,0,[-1]*l)
    return best_score, best_translation

def induced_subgraph(graph, central_vertex): 
    _list = [central_vertex]
    direct_connections = np.where(graph[central_vertex] > 0)[0]
    random.shuffle(direct_connections)
    _list.extend(direct_connections)
    l = len(_list)
    graph_to_subgraph = dict()
    subgraph_to_graph = []
    count = 0
    for i in _list:
        graph_to_subgraph[i] = count
        subgraph_to_graph.append(i)
        count += 1
    for i in _list:
        for j, value in enumerate(graph[i]):
            if value == 1 and j not in subgraph_to_graph:
                graph_to_subgraph[j] = count
                subgraph_to_graph.append(j)
                count += 1
    return l, graph_to_subgraph, subgraph_to_graph.copy()

def identify_local_translations(arguments,graph):
    print
    res = [[] for i in range(len(graph[0]))]
    tables = [induced_subgraph(graph,i) for i in range(len(graph[0]))]
    for i in tqdm.tqdm(range(len(graph[0]))):
        l, gts, stg = tables[i]
        graph_copy = graph.copy()
        for j in range(1,l):
            best_cost, best_translation = search(arguments,graph,graph_copy,l,gts,stg)
            res[i].append([best_cost,best_translation])
            for k in range(len(best_translation)):
                if best_translation[k] > -1:
                    graph_copy[stg[k]][best_translation[k]] = 0 
    return res, tables

def propagate_central_pattern(graph,central_vertex,local_translations,tables):
    l, gts, stg = induced_subgraph(graph,central_vertex)
    patterns = [[-1]*l for i in range(len(graph))]
    costs = [10000]*len(graph[0])
    for i in range(l):
        patterns[central_vertex][i] = stg[i]
    costs[central_vertex] = 0
    while True:
        print(sum(costs))
        modifications = False
        for line in range(len(tables)):
            _,gts,_ = tables[line]
            for cost,translation in local_translations[line]:
                new_cost = cost + costs[line]
                if new_cost < costs[translation[0]]:
                    modifications = True
                    for reference,assignation in enumerate(patterns[line]):
                        new_assignation = -1 if assignation == -1 else gts[assignation] 
                        patterns[translation[0]][reference] = -1 if new_assignation == -1 else translation[new_assignation] 
                    costs[translation[0]] = new_cost
        if not modifications:
            print("finished propagating")
            break
    return np.array(patterns),costs

def discover_global_translations(graph,central_vertex,patterns):
    patterns = np.array(patterns)
    l, gts, stg = induced_subgraph(graph,central_vertex)
    global_translations = [[-1]*len(graph[0]) for i in range(len(graph[0]))]
    costs = [len(graph[0])]*len(graph[0])
    for i in range(l):
        global_translations[central_vertex][i] = patterns[central_vertex][i]
    other_values = np.arange(len(graph[0]))
    final = np.setdiff1d(other_values,patterns[central_vertex])
    np.random.shuffle(final)
    for j in range(0,len(final)):
        global_translations[central_vertex][j+i+1] = final[j]
    costs[central_vertex] = 0
    nodes_to_spread = deque([central_vertex])
    while True:
        try:
            random.shuffle(nodes_to_spread)
            print(sum(costs))
            node = int(nodes_to_spread.popleft())
            local_translations = patterns[node]
            for index_translation, translated_node in enumerate(local_translations):
                if index_translation == 0 or translated_node == -1:
                    continue
                new_line = np.zeros(len(graph),dtype=np.int32)-1
                for i in range(len(graph)):
                    new_value = global_translations[node][i]
                    new_line[i] = patterns[new_value][index_translation]
                new_cost = len(graph[0])-np.unique(new_line).shape[0]
                cost = costs[translated_node]
                if new_cost < cost:
                    global_translations[translated_node] = new_line
                    nodes_to_spread.extend([translated_node])
                    costs[translated_node] = new_cost
        except IndexError:
            break
    return np.array(global_translations),costs

def revive_nodes(global_translations,max_cost,costs,patterns):
    for index,cost in enumerate(costs):
        if cost == max_cost:
            print(index,cost)
            for index2,value in enumerate(patterns[index]):
                global_translations[index][index2] = value

def translations_to_dict(translations):
    dict_translations = [dict() for i in range(translations.shape[0])]
    for index_pattern,pattern in enumerate(translations):
        for index_translation,node in enumerate(pattern,start=1):
            if node > -1:
                dict_translations[index_pattern][node] = index_translation
    return dict_translations

def convert_list_dict_to_string(list_dict):
    return list_dict.__str__().replace("},","},\n")

class Get_Hops(object):
    def __init__(self,graph):
        self.graph = graph
    
    @functools.lru_cache(maxsize=1024)
    def get_x_hop(self,node,x):
        if x == 0:
            return set([node])
        else:
            direct_vertexes = np.where(self.graph[node] > 0)[0]
            nodes = set()
            for direct_vertex in direct_vertexes:
                 nodes |= self.get_x_hop(direct_vertex,x-1)
            return nodes

def discover_stride_nodes(graph,central_vertex,hops_keep=2,hops_die=[1],dead_nodes = set()):
    alive_nodes = set([central_vertex])
    central_list = [central_vertex]
    dead_nodes = dead_nodes.copy()
    get_hops = Get_Hops(graph)
    while len(central_list) > 0:
        list_to_die = set()
        for new_central in central_list:
            for hops_die_index in hops_die:
                list_to_die |= get_hops.get_x_hop(new_central,hops_die_index)
        list_to_die = list(list_to_die)
        for node in list_to_die:
            if node not in alive_nodes:
                dead_nodes |= set([node])


        list_to_live = set()
        for new_central in central_list:
            list_to_live |= get_hops.get_x_hop(new_central,hops_keep)            
        list_to_live = list(list_to_live)

        central_list = list()
        for node in list_to_live:
            if node not in dead_nodes:
                if node not in alive_nodes:
                    alive_nodes |= set([node])
                    if node not in central_list:
                        central_list.append(node)
    return alive_nodes,dead_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translation search')
    parser.add_argument('--graph', '-g', help='graph file'     , required=True, type=str)
    parser.add_argument('--name' , '-n', help='output name'    , required=True, type=str)
    parser.add_argument('--alpha' , '-a', help='alfa parameter' , default=1.0  , type=float)
    parser.add_argument('--beta' , '-b', help='beta parameter' , default=100.0, type=float)
    parser.add_argument('--gamma', '-c', help='gamma parameter', default=100.0, type=float)
    parser.add_argument('--delta', '-d', help='delta parameter', default=100.0, type=float)
    parser.add_argument('--strides', '-s', help='strides', action='store_true')

    args = parser.parse_args()

    if args.graph == "grid":
        graph = generate_grid_graph(32,32)
    else:
        graph = read_graph(args.graph)

    print("Finding the center of the graph")
    best_distance, center_node = get_center(graph)
    print("Identifying local translations")
    local_translations, tables = identify_local_translations(args,graph)
    print("Propagating local translations")
    patterns, costs = propagate_central_pattern(graph,center_node,local_translations,tables)
    print("Creating global translations")
    global_translations, costs = discover_global_translations(graph,center_node,patterns)
    revive_nodes(global_translations,len(graph[0]),costs,patterns=patterns)
    translations = translations_to_dict(patterns)

    f = open("translations/{}".format(args.name),"w")
    f.write(convert_list_dict_to_string(translations))
    f.close()
    get_hops = Get_Hops(graph)

    print("Generating strides")

    alive, dead = discover_stride_nodes(graph,center_node)
    if args.strides:
        alive2,dead2 = discover_stride_nodes(graph,center_node,hops_die=[1,2,3],hops_keep=4,dead_nodes=dead)
        alive3,dead3 = discover_stride_nodes(graph,center_node,hops_die=[1,2,3,4,5,6,7],hops_keep=8,dead_nodes=dead2)

    print("Stride 1, Alive/Dead/All",len(alive),len(dead),len(alive)+len(dead))
    if args.strides:
        print("Stride 2, Alive/Dead/All",len(alive2),len(dead2),len(alive2)+len(dead2))
        print("Stride 3, Alive/Dead/All",len(alive3),len(dead3),len(alive3)+len(dead3))

    one_hops_center = get_hops.get_x_hop(center_node,1) | set([center_node])
    two_hops_center = get_hops.get_x_hop(center_node,2) | set([center_node])
    if args.strides:
        four_hops_center = get_hops.get_x_hop(center_node,4) | set([center_node])
        eight_hops_center = get_hops.get_x_hop(center_node,8) | set([center_node])

    alive_two_hops = two_hops_center - set(dead) 
    if args.strides:
        alive_four_hops = four_hops_center - set(dead2) 
        alive_eight_hops = eight_hops_center - set(dead3) 

    one_hop_translations_indexes = sorted([np.where(np.array(global_translations[center_node]) == node)[0][0] for node in one_hops_center])
    two_hop_translations_indexes = sorted([np.where(np.array(global_translations[center_node]) == node)[0][0] for node in alive_two_hops])
    if args.strides:
        four_hop_translations_indexes = sorted([np.where(np.array(global_translations[center_node]) == node)[0][0] for node in alive_four_hops])
        eight_hop_translations_indexes = sorted([np.where(np.array(global_translations[center_node]) == node)[0][0] for node in alive_eight_hops])

    convolution_filters_no_stride = np.array(global_translations)[:,one_hop_translations_indexes]

    alive_list = list(alive)
    translations_one_stride = np.array(global_translations)[alive_list]
    for i in range (translations_one_stride.shape[0]):
        for j in range(translations_one_stride.shape[1]):
            try:
                translations_one_stride[i,j] = alive_list.index(translations_one_stride[i,j])
            except ValueError as e:
                translations_one_stride[i,j] = -1
    convolution_filters_one_stride = translations_one_stride[:,two_hop_translations_indexes]

    if args.strides:

        alive_list_2 = [alive_list.index(x) for x in alive2]
        translations_two_stride = translations_one_stride[alive_list_2]

        for i in range (translations_two_stride.shape[0]):
            for j in range(translations_two_stride.shape[1]):
                try:
                    translations_two_stride[i,j] = alive_list_2.index(translations_two_stride[i,j])
                except ValueError as e:
                    translations_two_stride[i,j] = -1
        convolution_filters_two_stride = translations_two_stride[:,four_hop_translations_indexes]

        alive_list_3 = [alive_list_2.index(alive_list.index(x)) for x in alive3]
        translations_four_stride = translations_two_stride[alive_list_3]

        for i in range (translations_four_stride.shape[0]):
            for j in range(translations_four_stride.shape[1]):
                try:
                    translations_four_stride[i,j] = alive_list_3.index(translations_four_stride[i,j])
                except ValueError as e:
                    translations_four_stride[i,j] = -1
        convolution_filters_four_stride = translations_four_stride[:,eight_hop_translations_indexes]

    no_stride = dict(alive=np.arange(len(graph[0])), translations=translations_to_dict(convolution_filters_no_stride))
    one_stride = dict(alive=alive_list, translations=translations_to_dict(convolution_filters_one_stride))
    if args.strides:
        two_stride = dict(alive=alive_list_2, translations=translations_to_dict(convolution_filters_two_stride))
        three_stride = dict(alive=alive_list_3, translations=translations_to_dict(convolution_filters_four_stride))

    if args.strides:
        translations = {
            0:no_stride,
            1:one_stride,
            2:two_stride,
            3:three_stride}
    else:
        translations = {
            0:no_stride,
            1:one_stride,}


    f = open("translations/{}.pkl".format(args.name),"wb")
    pickle.dump(translations,f)
    f.close()
