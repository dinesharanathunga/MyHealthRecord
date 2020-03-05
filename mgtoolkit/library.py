from properties import resources
from exception import MetagraphException
from numpy import matrix
#from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString, Point
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
from enums import Ipv4ProtocolNumbers

def singleton(cls):
    """A helper function to ease implementing singletons.
    This should be used as a decorator to the
    class that should be a singleton.
    :param cls: class that should be a singleton
    :return: singleton instance of the class
    """
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance

class Triple(object):
    """ Captures a set of co-inputs, co-outputs and edges between two metagraph elements.
    """

    def __init__(self, coinputs, cooutputs, edges):
        if edges is None:
            raise MetagraphException('edges', resources['value_null'])

        self.coinputs = coinputs
        self.cooutputs = cooutputs
        self.edges = edges

    def coinputs(self):
        """ The co-inputs of the Triple object
        :return: set
        """
        return self.coinputs

    def cooutputs(self):
        """ The co-outputs of the Triple object
        :return: set
        """
        return self.cooutputs

    def edges(self):
        """ The edges of the Triple object
        :return: set
        """
        return self.edges

    def __repr__(self):
        if isinstance(self.edges, list):
            edge_desc = [repr(edge) for edge in self.edges]
        else:
            edge_desc = [repr(self.edges)]
        full_desc = ''
        for desc in edge_desc:
            if full_desc == '':
                full_desc = desc
            else:
                full_desc += ', ' + desc
        return 'Triple(%s, %s, %s)' % (self.coinputs, self.cooutputs, full_desc)

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other,Triple):
            return False

        return (self.coinputs == other.coinputs and
                self.cooutputs == other.cooutputs and
                len(self.edges) == len(other.edges) and
                self.edges == other.edges)

class Node(object):
    """ Represents a metagraph node.
    """

    def __init__(self, element_set):
        if element_set is None or len(element_set) == 0:
            raise MetagraphException('element_set', resources['value_null'])
        if not isinstance(element_set, set):
            raise MetagraphException('element_set', resources['format_invalid'])

        self.element_set = element_set

    def get_element_set(self):
        """ Returns the node elements
        :return: set
        """
        return self.element_set

    def __repr__(self):
        return 'Node(%s)' % self.element_set

class Edge(object):
    """ Represents a metagraph edge.
    """

    def __init__(self, invertex, outvertex, attributes=None, label=None):
        if invertex is None or len(invertex) == 0:
            raise MetagraphException('invertex', resources['value_null'])
        if outvertex is None or len(outvertex) == 0:
            raise MetagraphException('outvertex', resources['value_null'])
        if not isinstance(invertex, set):
            raise MetagraphException('invertex', resources['format_invalid'])
        if not isinstance(outvertex, set):
            raise MetagraphException('outvertex', resources['format_invalid'])

        self.invertex = invertex
        self.outvertex = outvertex
        self.attributes = attributes
        self.label = label

        # include attributes as part if invertex
        if attributes is not None:
            invertex = list(self.invertex)
            for attribute in attributes:
                if attribute not in invertex:
                    invertex.append(attribute)
            self.invertex = set(invertex)

    def __repr__(self):
        return 'Edge(%s, %s)' % (self.invertex, self.outvertex)

    def invertex(self):
        """ Returns the invertex of the edge.
        :return: set
        """
        return self.invertex

    def outvertex(self):
        """ Returns the outvertex of the edge.
        :return: set
        """
        return self.outvertex

    def label(self):
        """ Returns the label of the edge.
        :return: string
        """
        return self.label

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, Edge):
            return False

        return (self.invertex == other.invertex and
                self.outvertex == other.outvertex and
                self.attributes == other.attributes)

class Metapath(object):
    """ Represents a metapath between a source and a target node in a metagraph.
    """

    def __init__(self, source, target, edge_list):
        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        self.source = source
        self.target = target
        self.edge_list = edge_list

    def source(self):
        """ Returns the source of the metapath.
        :return: set
        """
        return self.source

    def target(self):
        """ Returns the target of the metapath.
        :return: set
        """
        return self.target

    def edge_list(self):
        """ Returns the list of edges of the metapath.
        :return: set
        """
        return self.edge_list

    def __repr__(self):
        edge_desc = [repr(edge) for edge in self.edge_list]
        full_desc = 'source: %s, target: %s' % (self.source, self.target)
        for desc in edge_desc:
            if full_desc == '':
                full_desc = desc
            else:
                full_desc += ", " + desc
        return 'Metapath({ %s })' % full_desc

    def dominates(self, metapath):
        """Checks whether current metapath dominates that provided.
        :param metapath: Metapath object
        :return: boolean
        """
        if metapath is None:
            raise MetagraphException('metapath', resources['value_null'])

        input1 = self.source
        input2 = metapath.source

        output1 = self.target
        output2 = metapath.target

        if input1.issubset(input2) and output2.issubset(output1):
            return True

        return False

class Metagraph(object):
    """ Represents a metagraph.
    """

    def __init__(self, generator_set):
        if generator_set is None or len(generator_set) == 0:
            raise MetagraphException('generator_set', resources['value_null'])
        if not isinstance(generator_set, set):
            raise MetagraphException('generator_set', resources['format_invalid'])

        self.nodes = []
        self.edges = []
        self.generating_set = generator_set
        self.a_star = None

    def add_node(self, node):
        """ Adds a node to the metagraph.
        :param node: Node object
        :return: None
        """
        if not isinstance(node, Node):
            raise MetagraphException('node', resources['format_invalid'])

        # nodes cant be null or empty
        if node is None:
            raise MetagraphException('node', resources['value_null'])

        # each element in node must be in the generating set
        not_found = [element not in self.generating_set for element in node.element_set]
        if True in not_found:
            raise MetagraphException('node', resources['range_invalid'])

        if not MetagraphHelper().is_node_in_list(node, self.nodes):
            self.nodes.append(node)

    def remove_node(self, node):
        """ Removes a specified node from the metagraph.
        :param node: Node object
        :return: None
        """

        if not isinstance(node, Node):
            raise MetagraphException('node', resources['format_invalid'])

        # nodes cant be null or empty
        if node is None:
            raise MetagraphException('node', resources['value_null'])

        if not MetagraphHelper().is_node_in_list(node, self.nodes):
            raise MetagraphException('node', resources['value_not_found'])

        self.nodes.remove(node)

    def add_nodes_from(self, nodes_list):
        """ Adds nodes from the given list to the metagraph.
        :param nodes_list: list of Node objects
        :return: None
        """

        if nodes_list is None or len(nodes_list) == 0:
            raise MetagraphException('nodes_list', resources['value_null'])

        for node in nodes_list:
            if not isinstance(node, Node):
                raise MetagraphException('nodes_list', resources['format_invalid'])

        for node in nodes_list:
            if not MetagraphHelper().is_node_in_list(node, self.nodes):
                self.nodes.append(node)

    def remove_nodes_from(self, nodes_list):
        """ Removes nodes from the given list from the metagraph.
        :param nodes_list: list of Node objects
        :return: None
        """

        if nodes_list is None or len(nodes_list) == 0:
                raise MetagraphException('nodes_list', resources['value_null'])

        for node in nodes_list:
            if not isinstance(node, set):
                raise MetagraphException('nodes_list', resources['format_invalid'])
            if not MetagraphHelper().is_node_in_list(node, self.nodes):
                raise MetagraphException('nodes_list', resources['value_not_found'])

        for node in nodes_list:
            self.nodes.remove(node)

    def add_edge(self, edge):
        """ Adds the given edge to the metagraph.
        :param edge: Edge object
        :return: None
        """

        if not isinstance(edge, Edge):
            raise MetagraphException('edge', resources['format_invalid'])

        # add to list of nodes first
        node1 = Node(edge.invertex)
        node2 = Node(edge.outvertex)
        if not MetagraphHelper().is_node_in_list(node1, self.nodes):
            self.nodes.append(node1)
        if not MetagraphHelper().is_node_in_list(node2, self.nodes):
            self.nodes.append(node2)

        #..then edges
        if not MetagraphHelper().is_edge_in_list(edge, self.edges):
            self.edges.append(edge)

    def remove_edge(self, edge):
        """ Removes the given edge from the metagraph.
        :param edge: Edge object
        :return:None
        """

        if not isinstance(edge, Edge):
            raise MetagraphException('edge', resources['format_invalid'])

        # remove edge
        if edge in self.edges:
            self.edges.remove(edge)

    def add_edges_from(self, edge_list):
        """ Adds the given list of edges to the metagraph.
        :param edge_list: list of Edge objects
        :return: None
        """

        if edge_list is None or len(edge_list) == 0:
            raise MetagraphException('edge_list', resources['value_null'])

        for edge in edge_list:
            if not isinstance(edge, Edge):
                raise MetagraphException('edge', resources['format_invalid'])

        for edge in edge_list:
            node1 = Node(edge.invertex)
            node2 = Node(edge.outvertex)
            if not MetagraphHelper().is_node_in_list(node1, self.nodes):
                self.nodes.append(node1)
            if not MetagraphHelper().is_node_in_list(node2, self.nodes):
                self.nodes.append(node2)
            if edge not in self.edges:
                self.edges.append(edge)

    def remove_edges_from(self, edge_list):
        """ Removes edges from the given list from the metagraph.
        :param edge_list: list of Edge objects
        :return: None
        """

        if edge_list is None or len(edge_list) == 0:
            raise MetagraphException('edge_list', resources['value_null'])

        for edge in edge_list:
            if not isinstance(edge, Edge):
                raise MetagraphException('edge', resources['format_invalid'])

        for edge in edge_list:
            if MetagraphHelper().is_edge_in_list(edge, self.edges):
                self.edges.remove(edge)

    def nodes(self):
        """ Returns a list of metagraph nodes.
        :return: list of Node objects
        """
        return self.nodes

    def edges(self):
        """ Returns a list of metagraph edges.
        :return: list of Edge objects.
        """
        return self.edges

    def get_edges(self, invertex, outvertex):
        """ Retrieves all edges between a given invertex and outvertex.
        :param invertex: set
        :param outvertex: set
        :return: list of Edge objects.
        """

        if invertex is None:
            raise MetagraphException('invertex', resources['value_null'])
        if outvertex is None:
            raise MetagraphException('outvertex', resources['value_null'])

        result = []
        if len(self.edges) > 0:
            for edge in self.edges:
                if (invertex.issubset(edge.invertex)) and (outvertex.issubset(edge.outvertex)) and (edge not in result):
                    result.append(edge)

        return result

    @staticmethod
    def get_coinputs(edge, x_i):
        """ Returns the set of co-inputs for element x_i in the given edge.
        :param edge: Edge object
        :param x_i: invertex element
        :return: set
        """

        coinputs = None
        all_inputs = edge.invertex
        if x_i in list(all_inputs):
            coinputs = list(all_inputs)
            coinputs.remove(x_i)
        if coinputs is not None and len(coinputs) > 0:
            return set(coinputs)
        return None

    @staticmethod
    def get_cooutputs(edge, x_j):
        """ Returns the set of co-outputs for element x_j in the given edge.
        :param edge: Edge object
        :param x_j: outvertex element
        :return: set
        """

        cooutputs = None
        all_outputs = edge.outvertex
        if x_j in list(all_outputs):
            cooutputs = list(all_outputs)
            cooutputs.remove(x_j)
        if cooutputs is not None and len(cooutputs) > 0:
            return set(cooutputs)
        return None

    def adjacency_matrix1(self):
        """ Returns the adjacency matrix of the metagraph.
        :return: numpy.matrix
        """
        # get matrix size
        size = len(self.generating_set)
        adj_matrix = MetagraphHelper().get_null_matrix(size, size)

        # one triple for each edge e connecting x_i to x_j
        count=1
        for i in range(size):
            for j in range(size):
                x_i = list(self.generating_set)[i]
                x_j = list(self.generating_set)[j]
                # multiple edges may exist between x_i and x_j
                edges = self.get_edges({x_i}, {x_j})
                if len(edges) > 0:
                   triples_list = []
                   for edge in edges:
                        coinputs = self.get_coinputs(edge, x_i)
                        cooutputs = self.get_cooutputs(edge, x_j)
                        triple = Triple(coinputs, cooutputs, edge)
                        if triple not in triples_list:
                           triples_list.append(triple)

                   adj_matrix[i][j] = triples_list
                print('count= %s'%count)
                count +=1

        # return adj_matrix
        # noinspection PyCallingNonCallable
        return matrix(adj_matrix)

    def adjacency_matrix(self):
        """ Returns the adjacency matrix of the metagraph.
        :return: numpy.matrix
        """
        # get matrix size
        size = len(self.generating_set)
        adj_matrix = MetagraphHelper().get_null_matrix(size, size)

        # create lookup table
        #print('here1')
        count=1
        triples_lookup=dict()
        for edge in self.edges:
             for elt1 in edge.invertex:
                  for elt2 in edge.outvertex:
                      coinputs = self.get_coinputs(edge, elt1)
                      cooutputs = self.get_cooutputs(edge, elt2)
                      triple = Triple(coinputs, cooutputs, edge)
                      if (elt1,elt2) not in triples_lookup:
                         triples_lookup[(elt1,elt2)] = []
                      triples_lookup[(elt1,elt2)].append(triple)
                      #print('count1=%s'%count)
                      count+=1

        #print('here2')
        count=1
        gen_elts = list(self.generating_set)
        for i in range(size):
             for j in range(size):
                  x_i = gen_elts[i]
                  x_j = gen_elts[j]
                  try:
                      adj_matrix[i][j] = triples_lookup[(x_i,x_j)]
                  except BaseException,e:
                      pass
             #print('count2=%s'%count)
             #count+=1

        # return adj_matrix
        # noinspection PyCallingNonCallable
        return matrix(adj_matrix)

    def equivalent(self, metagraph2):
        """Checks if current metagraph is equivalent to the metagraph provided.
        :param metagraph2: Metagraph object
        :return: boolean
        """

        if metagraph2 is None:
            raise MetagraphException('metagraph2', resources['value_null'])

        if self.dominates(metagraph2) and metagraph2.dominates(self):
            return True

        return False

    def add_metagraph(self, metagraph2):
        """ Adds the given metagraph to current and returns the composed result.
        :param metagraph2: Metagraph object
        :return: Metagraph object
        """

        if metagraph2 is None:
            raise MetagraphException('metagraph2', resources['value_null'])

        generating_set1 = self.generating_set
        generating_set2 = metagraph2.generating_set

        if generating_set2 is None or len(generating_set2) == 0:
            raise MetagraphException('metagraph2.generating_set', resources['value_null'])

        # check if the generating sets of the matrices overlap (otherwise no sense in combining metagraphs)
        #intersection=generating_set1.intersection(generating_set2)
        #if intersection==None:
        #    raise MetagraphException('generating_sets', resources['no_overlap'])

        if len(generating_set1.difference(generating_set2)) == 0 and \
           len(generating_set2.difference(generating_set1)) == 0:
            # generating sets are identical..simply add edges
            # size = len(generating_set1)
            for edge in metagraph2.edges:
                if edge not in self.edges:
                    self.add_edge(edge)
        else:
            # generating sets overlap but are different...combine generating sets and then add edges
            # combined_generating_set = generating_set1.union(generating_set2)
            self.generating_set = generating_set1.union(generating_set2)
            for edge in metagraph2.edges:
                if edge not in self.edges:
                    self.add_edge(edge)

        return self

    def multiply_metagraph(self, metagraph2):
        """ Multiplies the metagraph with that provided and returns the result.
        :param metagraph2: Metagraph object
        :return: Metagraph object
        """

        if metagraph2 is None:
            raise MetagraphException('metagraph2', resources['value_null'])

        generating_set1 = self.generating_set
        generating_set2 = metagraph2.generating_set

        if generating_set2 is None or len(generating_set2) == 0:
            raise MetagraphException('metagraph2.generator_set', resources['value_null'])

        # check generating sets are identical
        if not(len(generating_set1.difference(generating_set2)) == 0 and
           len(generating_set2.difference(generating_set1)) == 0):
            raise MetagraphException('generator_sets', resources['not_identical'])

        adjacency_matrix1 = self.adjacency_matrix().tolist()
        adjacency_matrix2 = metagraph2.adjacency_matrix().tolist()
        size = len(generating_set1)
        resultant_adjacency_matrix = MetagraphHelper().get_null_matrix(size, size)

        for i in range(size):
            for j in range(size):
                resultant_adjacency_matrix[i][j] = MetagraphHelper().multiply_components(
                    adjacency_matrix1, adjacency_matrix2, generating_set1, i, j, size)

        # extract new edge list
        new_edge_list = MetagraphHelper().get_edges_in_matrix(resultant_adjacency_matrix, self.generating_set)
        # clear current edge list and append new
        self.edges = []
        if len(new_edge_list) > 0:
            self.add_edges_from(new_edge_list)

        return self

    def get_closure(self, iterations=1):#0
        """ Returns the closure matrix (i.e., A*) of the metagraph.
        :return: numpy.matrix
        """

        # gen set size N
        # edge count C
        # proj elts M

        #print('get adjacency_matrix')
        adjacency_matrix = self.adjacency_matrix().tolist()

        i = 0
        size = len(self.generating_set)
        a = dict()
        a[i] = adjacency_matrix
        a_star = adjacency_matrix

        #TODO: re-enable
        if iterations==0:
            iterations = size
        #print('iterations:%s'%iterations)

        # O(N4C2)
        for i in range(iterations):
            #print(' iteration %s --------------'%i)
            # O(N3C2)
            #print('multiply_adjacency_matrices')
            a[i+1] = MetagraphHelper().multiply_adjacency_matrices(a[i],
                                                                   self.generating_set,
                                                                   adjacency_matrix,
                                                                   self.generating_set)
            #print('add_adjacency_matrices')
            a_star = MetagraphHelper().add_adjacency_matrices(a_star,
                                                              self.generating_set,
                                                              a[i+1],
                                                              self.generating_set)
            #print('add_adjacency_matrices complete')
            if a[i+1] == a[i]:
                break

        # noinspection PyCallingNonCallable
        return matrix(a_star)

    def get_all_metapaths_from2(self, source, target, props=None):
        """ Retrieves all metapaths between given source and target in the metagraph.
        :param source: set
        :param target: set
        :return: list of Metapath objects
        """

        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        # check subset
        if not source.intersection(self.generating_set) == source:
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.intersection(self.generating_set) == target:
            raise MetagraphException('target', resources['not_a_subset'])

        # compute A* first
        if self.a_star is None:
            #print('computing closure..')
            self.a_star = self.get_closure().tolist()
            #print('closure computation- %s'%(time.time()- start))

        metapaths = []
        all_applicable_input_rows = []
        for x_i in source:
            index = list(self.generating_set).index(x_i)
            if index not in all_applicable_input_rows:
                all_applicable_input_rows.append(index)

        cumulative_output_global = []
        cumulative_edges_global = []
        for i in all_applicable_input_rows:
            mp_exist_for_row=False
            cumulative_output_local = []
            cumulative_edges_local = []
            for x_j in target:
                j = list(self.generating_set).index(x_j)

                if self.a_star[i][j] is not None:
                    mp_exist_for_row = True

                    # x_j is already an output
                    cumulative_output_local.append(x_j)
                    triples = MetagraphHelper().get_triples(self.a_star[i][j])
                    for triple in triples:
                        # retain cooutputs
                        output = triple.cooutputs
                        if output is not None:
                            cumulative_output_local += output
                        if output is not None:
                            cumulative_output_global += output

                        #... and edges
                        if isinstance(triple.edges, Edge):
                            edges = MetagraphHelper().get_edge_list([triple.edges])
                        else:
                            edges = MetagraphHelper().get_edge_list(triple.edges)

                        for edge in edges:
                            if edge not in cumulative_edges_local:
                                cumulative_edges_local.append(edge)
                            if edge not in cumulative_edges_global:
                                cumulative_edges_global.append(edge)

            if not mp_exist_for_row:
               continue

            # check if cumulative outputs form a cover for the target
            if set(target).issubset(set(cumulative_output_local)):
                if set(cumulative_edges_local) not in metapaths:
                    metapaths.append(set(cumulative_edges_local))

            elif set(target).issubset(set(cumulative_edges_global)):
                if set(cumulative_edges_global) not in metapaths:
                    metapaths.append(set(cumulative_edges_global))

            #else:
            #    break

        if len(metapaths)>0:
            #result=[]
            #for path in metapaths:
            #    mp = Metapath(source, target, list(path))
            #    result.append(mp)
            #return result

            if True:
                valid_metapaths = []
                processed_edge_lists=[]
                from itertools import combinations
                for metapath in metapaths:
                    all_subsets = sum(map(lambda r: list(combinations(list(metapath), r)), range(1, len(list(metapath))+1)), [])
                    for path in all_subsets:
                        if len(path) <= len(metapath): # metapaths
                            edge_list2 = self.get_edge_list2(path)
                            if len(processed_edge_lists)>0:
                                if MetagraphHelper().is_edge_list_included(edge_list2,processed_edge_lists):
                                    continue
                            if props is not None:
                                mp = Metapath(source.union(props), target, edge_list2)
                            else:
                                mp = Metapath(source, target, edge_list2)
                            if self.is_metapath(mp): # is_edge_dominant_metapath
                               valid_metapaths.append(mp)
                return valid_metapaths

        return None

        ''' NEW
        if len(metapaths)>0:
            result=[]
            for path in metapaths:
                mp = Metapath(source, target, list(path))
                if self.is_metapath(mp):
                   result.append(mp)
            return result
        else:
            return None'''

        ''' OLD
        valid_metapaths = []
        from itertools import combinations
        all_subsets = sum(map(lambda r: list(combinations(metapaths, r)), range(1, len(metapaths)+1)), [])
        for path in all_subsets:
            if len(path) <= len(metapaths):
                mp = Metapath(source, target, list(path))
                if self.is_metapath(mp):
                    valid_metapaths.append(mp)

        if len(valid_metapaths)>0:
            return valid_metapaths

        return None'''

    def get_all_metapaths_from200(self, source, target):
        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        # check subset
        if not source.intersection(self.generating_set) == source:
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.intersection(self.generating_set) == target:
            raise MetagraphException('target', resources['not_a_subset'])

        # compute A* first
        if self.a_star is None:
            print('computing closure..')
            self.a_star = self.get_closure().tolist()
            #print('closure computation complete')
            #print('closure computation- %s'%(time.time()- start))

        print('find mps')
        metapaths = []
        all_applicable_input_rows = []
        for x_i in source:
            index = list(self.generating_set).index(x_i)
            if index not in all_applicable_input_rows:
                all_applicable_input_rows.append(index)

        cumulative_output_global = []
        cumulative_edges_global = []
        for j in all_applicable_input_rows:
            mp_exists=False
            for i in all_applicable_input_rows:
                if self.a_star[i][j] is not None:
                    # metapath from {x_i | i in I} to C exists
                    mp_exists = True
                    break
            if not mp_exists:
               # TODO: return to step1 and repeat with another set of rows
               return None

        metapaths=[]
        for x_j in target:
            j = list(self.generating_set).index(x_j)
            triples_set=set()
            for i in all_applicable_input_rows:
                triples = MetagraphHelper().get_triples(self.a_star[i][j])
                triples_set = triples_set.union(set(triples))
                if MetagraphHelper().forms_cover(triples_set, target):
                    metapath = MetagraphHelper().get_metapath_from_triples(triples_set)
                    metapaths.append(metapath)

        for i in all_applicable_input_rows:
            triples_set=set()
            for x_j in target:
                j = list(self.generating_set).index(x_j)
                triples = MetagraphHelper().get_triples(self.a_star[i][j])
                triples_set = triples_set.union(set(triples))
                if MetagraphHelper().forms_cover(triples_set, target):
                    metapath = MetagraphHelper().get_metapath_from_triples(triples_set)
                    metapaths.append(metapath)

        return metapaths

    def get_all_metapaths_from100(self, source, target):
        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        # check subset
        if not source.intersection(self.generating_set) == source:
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.intersection(self.generating_set) == target:
            raise MetagraphException('target', resources['not_a_subset'])

        # compute A* first
        if self.a_star is None:
            print('computing closure..')
            self.a_star = self.get_closure().tolist()
            #print('closure computation complete')
            #print('closure computation- %s'%(time.time()- start))

        print('find mps')
        metapaths = []
        all_applicable_input_rows = []
        for x_i in source:
            index = list(self.generating_set).index(x_i)
            if index not in all_applicable_input_rows:
                all_applicable_input_rows.append(index)

        cumulative_output_global = []
        cumulative_edges_global = []
        for i in all_applicable_input_rows:
            mp_exist_for_row=False
            cumulative_output_local = []
            cumulative_edges_local = []
            for x_j in target:
                j = list(self.generating_set).index(x_j)

                if self.a_star[i][j] is not None:
                    mp_exist_for_row = True

                    # x_j is already an output
                    cumulative_output_local.append(x_j)
                    triples = MetagraphHelper().get_triples(self.a_star[i][j])
                    for triple in triples:
                        # retain cooutputs
                        output = triple.cooutputs
                        if output is not None:
                            cumulative_output_local += output
                        if output is not None:
                            cumulative_output_global += output

                        #... and edges
                        if isinstance(triple.edges, Edge):
                            edges = MetagraphHelper().get_edge_list([triple.edges])
                        else:
                            edges = MetagraphHelper().get_edge_list(triple.edges)

                        for edge in edges:
                            if edge not in cumulative_edges_local:
                                cumulative_edges_local.append(edge)
                            if edge not in cumulative_edges_global:
                                cumulative_edges_global.append(edge)

            if not mp_exist_for_row:
               continue

            # check if cumulative outputs form a cover for the target
            if set(target).issubset(set(cumulative_output_local)):
                if set(cumulative_edges_local) not in metapaths:
                    metapaths.append(set(cumulative_edges_local))

            elif set(target).issubset(set(cumulative_edges_global)):
                if set(cumulative_edges_global) not in metapaths:
                    metapaths.append(set(cumulative_edges_global))

            #else:
            #    break

        print('check result')
        from itertools import combinations
        if len(metapaths)>0:
            result=[]
            for path in metapaths:
                # all_subsets = sum(map(lambda r: list(combinations(list(path), r)), range(1, len(list(path))+1)), [])
                all_subsets = []
                print('path= %s'%path)
                i=1
                for p in all_subsets:
                    print('counter=%s'%i)
                    i+=1
                #for path2 in all_subsets:
                #    if len(path2) <= len(path):
                #        mp = Metapath(source, target, list(path2))
                #        if self.is_metapath(mp): # is_edge_dominant_metapath
                #           result.append(mp)


                # check no loops
                #if self.check_no_loops(list(path)):
                #    mp = Metapath(source, target, list(path))
                #    #print('mp src= %s, target= %s'%(source,target))
                #    #if self.is_metapath(mp):
                #    result.append(mp)
                    # TODO:compare this with coursex alg
            return result
        else:
            return None

    def get_all_metapaths_from(self, source, target, props=None):
        """ Retrieves all metapaths between given source and target in the metagraph.
        :param source: set
        :param target: set
        :return: list of Metapath objects
        """

        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        # check subset
        if not source.intersection(self.generating_set) == source:
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.intersection(self.generating_set) == target:
            raise MetagraphException('target', resources['not_a_subset'])

        # compute A* first
        if self.a_star is None:
            #print('computing closure..')
            self.a_star = self.get_closure().tolist()
            #print('closure computation- %s'%(time.time()- start))

        metapaths = []
        all_applicable_input_rows = []
        for x_i in source:
            index = list(self.generating_set).index(x_i)
            if index not in all_applicable_input_rows:
                all_applicable_input_rows.append(index)

        cumulative_output_global = []
        cumulative_edges_global = []
        for i in all_applicable_input_rows:
            mp_exist_for_row=False
            cumulative_output_local = []
            cumulative_edges_local = []
            for x_j in target:
                j = list(self.generating_set).index(x_j)

                if self.a_star[i][j] is not None:
                    mp_exist_for_row = True

                    # x_j is already an output
                    cumulative_output_local.append(x_j)
                    triples = MetagraphHelper().get_triples(self.a_star[i][j])
                    for triple in triples:
                        # retain cooutputs
                        output = triple.cooutputs
                        if output is not None:
                            cumulative_output_local += output
                        if output is not None:
                            cumulative_output_global += output

                        #... and edges
                        if isinstance(triple.edges, Edge):
                            edges = MetagraphHelper().get_edge_list([triple.edges])
                        else:
                            edges = MetagraphHelper().get_edge_list(triple.edges)

                        for edge in edges:
                            if edge not in cumulative_edges_local:
                                cumulative_edges_local.append(edge)
                            if edge not in cumulative_edges_global:
                                cumulative_edges_global.append(edge)

            if not mp_exist_for_row:
               continue

            # check if cumulative outputs form a cover for the target
            if set(target).issubset(set(cumulative_output_local)):
                if set(cumulative_edges_local) not in metapaths:
                    metapaths.append(set(cumulative_edges_local))

            elif set(target).issubset(set(cumulative_edges_global)):
                if set(cumulative_edges_global) not in metapaths:
                    metapaths.append(set(cumulative_edges_global))

            #else:
            #    break

        if len(metapaths)>0:
            #result=[]
            #for path in metapaths:
            #    mp = Metapath(source, target, list(path))
            #    result.append(mp)
            #return result

            if True:
                valid_metapaths = []
                processed_edge_lists=[]
                from itertools import combinations
                for metapath in metapaths:
                    if len(metapath)>25:
                        continue
                    all_subsets = sum(map(lambda r: list(combinations(list(metapath), r)), range(1, len(list(metapath))+1)), [])
                    for path in all_subsets:
                        if len(path) <= len(metapath): # metapaths
                            edge_list2 = self.get_edge_list2(path)
                            # TODO: enable
                            #if len(processed_edge_lists)>0:
                            #    if MetagraphHelper().is_edge_list_included(edge_list2,processed_edge_lists):
                            #        continue
                            if props is not None:
                                mp = Metapath(source.union(props), target, edge_list2)
                            else:
                                mp = Metapath(source, target, edge_list2)
                            if self.is_metapath(mp): # is_edge_dominant_metapath
                               valid_metapaths.append(mp)

                return valid_metapaths

        return None

        ''' NEW
        if len(metapaths)>0:
            result=[]
            for path in metapaths:
                mp = Metapath(source, target, list(path))
                if self.is_metapath(mp):
                   result.append(mp)
            return result
        else:
            return None'''

        ''' OLD
        valid_metapaths = []
        from itertools import combinations
        all_subsets = sum(map(lambda r: list(combinations(metapaths, r)), range(1, len(metapaths)+1)), [])
        for path in all_subsets:
            if len(path) <= len(metapaths):
                mp = Metapath(source, target, list(path))
                if self.is_metapath(mp):
                    valid_metapaths.append(mp)

        if len(valid_metapaths)>0:
            return valid_metapaths

        return None'''

    def get_all_metapaths_from_2000(self, source, target):
        """ Retrieves all metapaths between given source and target in the metagraph.
        :param source: set
        :param target: set
        :return: list of Metapath objects
        """

        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        # check subset
        if not source.intersection(self.generating_set) == source:
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.intersection(self.generating_set) == target:
            raise MetagraphException('target', resources['not_a_subset'])

        # compute A* first
        if self.a_star is None:
            #print('computing closure..')
            self.a_star = self.get_closure().tolist()
            #print('closure computation- %s'%(time.time()- start))

        metapaths = []
        all_applicable_input_rows = []
        for x_i in source:
            index = list(self.generating_set).index(x_i)
            if index not in all_applicable_input_rows:
                all_applicable_input_rows.append(index)

        cumulative_output_global = []
        cumulative_edges_global = []
        for i in all_applicable_input_rows:
            mp_exist_for_row=False
            cumulative_output_local = []
            cumulative_edges_local = []
            for x_j in target:
                j = list(self.generating_set).index(x_j)

                if self.a_star[i][j] is not None:
                    mp_exist_for_row = True

                    # x_j is already an output
                    cumulative_output_local.append(x_j)
                    triples = MetagraphHelper().get_triples(self.a_star[i][j])
                    for triple in triples:
                        # retain cooutputs
                        output = triple.cooutputs
                        if output is not None:
                            cumulative_output_local += output
                        if output is not None:
                            cumulative_output_global += output

                        #... and edges
                        if isinstance(triple.edges, Edge):
                            edges = MetagraphHelper().get_edge_list([triple.edges])
                        else:
                            edges = MetagraphHelper().get_edge_list(triple.edges)

                        for edge in edges:
                            if edge not in cumulative_edges_local:
                                cumulative_edges_local.append(edge)
                            if edge not in cumulative_edges_global:
                                cumulative_edges_global.append(edge)

            if not mp_exist_for_row:
               continue

            # check if cumulative outputs form a cover for the target
            if set(target).issubset(set(cumulative_output_local)):
                if set(cumulative_edges_local) not in metapaths:
                    metapaths.append(set(cumulative_edges_local))

            elif set(target).issubset(set(cumulative_edges_global)):
                if set(cumulative_edges_global) not in metapaths:
                    metapaths.append(set(cumulative_edges_global))

            else:
                break

        if len(metapaths)>0:
            #result=[]
            #for path in metapaths:
            #    mp = Metapath(source, target, list(path))
            #    result.append(mp)
            #return result

            valid_metapaths = []
            processed_edge_lists=[]
            from itertools import combinations
            for metapath in metapaths:
                if len(metapath)>25:
                   continue
                all_subsets = sum(map(lambda r: list(combinations(list(metapath), r)), range(1, len(list(metapath))+1)), [])
                for path in all_subsets:
                    if len(path) <= len(metapath): # metapaths
                        edge_list2 = self.get_edge_list2(path)
                        if len(processed_edge_lists)>0:
                            if MetagraphHelper().is_edge_list_included(edge_list2,processed_edge_lists):
                                continue
                        mp = Metapath(source, target, edge_list2)
                        if self.is_metapath(mp):
                            valid_metapaths.append(mp)
            return valid_metapaths

        return None

    def get_all_metapaths_from_600(self, source, target):
        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        # check subset
        if not source.intersection(self.generating_set) == source:
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.intersection(self.generating_set) == target:
            raise MetagraphException('target', resources['not_a_subset'])

        # compute A* first
        if self.a_star is None:
            print('computing closure..')
            self.a_star = self.get_closure().tolist()

        print('find mps')
        metapaths = []
        all_applicable_input_rows = []
        for x_i in source:
            index = list(self.generating_set).index(x_i)
            if index not in all_applicable_input_rows:
                all_applicable_input_rows.append(index)

        for x_j in target:
            mp_exists=False
            j = list(self.generating_set).index(x_j)
            for i in all_applicable_input_rows:
                if self.a_star[i][j] is not None:
                    mp_exists=True
                    break
        if not mp_exists:
            return None

        metapaths=[]
        for x_j in target:
            j = list(self.generating_set).index(x_j)
            triples_set=set()
            for i in all_applicable_input_rows:
                triples = MetagraphHelper().get_triples(self.a_star[i][j])
                triples_set = triples_set.union(set(triples))
                if MetagraphHelper().forms_cover(triples_set, target, x_j):
                    metapath = MetagraphHelper().get_metapath_from_triples(source, target, triples_set)
                    metapaths.append(metapath)

        for i in all_applicable_input_rows:
            triples_set=set()
            for x_j in target:
                j = list(self.generating_set).index(x_j)
                triples = MetagraphHelper().get_triples(self.a_star[i][j])
                triples_set = triples_set.union(set(triples))
                if MetagraphHelper().forms_cover(triples_set, target, x_j):
                    metapath = MetagraphHelper().get_metapath_from_triples(source, target, triples_set)
                    metapaths.append(metapath)


        for x_j in target:
            j = list(self.generating_set).index(x_j)
            for i in all_applicable_input_rows:
                triples_set=set()
                triples = MetagraphHelper().get_triples(self.a_star[i][j])
                triples_set = triples_set.union(set(triples))
                if MetagraphHelper().forms_cover(triples_set, target, x_j):
                    metapath = MetagraphHelper().get_metapath_from_triples(source, target, triples_set)
                    metapaths.append(metapath)

        for i in all_applicable_input_rows:
            for x_j in target:
                j = list(self.generating_set).index(x_j)
                triples_set=set()
                triples = MetagraphHelper().get_triples(self.a_star[i][j])
                triples_set = triples_set.union(set(triples))
                if MetagraphHelper().forms_cover(triples_set, target, x_j):
                    metapath = MetagraphHelper().get_metapath_from_triples(source, target, triples_set)
                    metapaths.append(metapath)

        triples_set=set()
        for i in all_applicable_input_rows:
            for x_j in target:
                j = list(self.generating_set).index(x_j)
                triples = MetagraphHelper().get_triples(self.a_star[i][j])
                triples_set = triples_set.union(set(triples))
                if MetagraphHelper().forms_cover(triples_set, target, x_j):
                    metapath = MetagraphHelper().get_metapath_from_triples(source, target, triples_set)
                    metapaths.append(metapath)

        triples_set=set()
        for x_j in target:
            j = list(self.generating_set).index(x_j)
            for i in all_applicable_input_rows:
                triples = MetagraphHelper().get_triples(self.a_star[i][j])
                triples_set = triples_set.union(set(triples))
                if MetagraphHelper().forms_cover(triples_set, target, x_j):
                    metapath = MetagraphHelper().get_metapath_from_triples(source, target, triples_set)
                    metapaths.append(metapath)

        #TODO : check for duplicates
        result=[]
        for mp in metapaths:
            if self.is_metapath(mp):
                result.append(mp)

        # A*[i][j]:   paths from x_i to x_j
        # metapaths from B to C - pure inputs/outputs: edges that connect exactly B or its subset to exactly C or its superset

        return result


    def check_no_loops(self, edges):
        import networkx as nx
        g = nx.DiGraph()
        for edge in edges:
            invertex=edge.invertex
            if edge.attributes is not None:
                invertex=edge.invertex.difference(edge.attributes)
            outvertex=edge.outvertex
            g.add_edge(frozenset(invertex), frozenset(outvertex))

        try:
            cycle = nx.simple_cycles(g)
            if len(cycle)>0:
                return False
            else:
                return True
        except BaseException, e:
            return True

    def get_edge_list(self, path):
        result=set()
        for elt in path:
            result = result.union(elt)

        return list(result)

    def get_edge_list2(self, path):
        result=set()
        for elt in path:
            result = result.union({elt})

        return list(result)

    def get_all_metapaths_from2(self, source, target):
        """ Retrieves all metapaths between given source and target in the metagraph.
        :param source: set
        :param target: set
        :return: list of Metapath objects
        """

        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        # check subset
        if not source.intersection(self.generating_set) == source:
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.intersection(self.generating_set) == target:
            raise MetagraphException('target', resources['not_a_subset'])

        # compute A* first
        if self.a_star is None:
            #print('computing closure..')
            self.a_star = self.get_closure().tolist()
            #print('closure computation- %s'%(time.time()- start))

        metapaths = []
        all_applicable_input_rows = []
        for x_i in source:
            index = list(self.generating_set).index(x_i)
            if index not in all_applicable_input_rows:
                all_applicable_input_rows.append(index)

        for i in all_applicable_input_rows:
            metapath_exist = False
            cumulative_output = []
            cumulative_edges = []
            for x_j in target:
                j = list(self.generating_set).index(x_j)
                if self.a_star[i][j] is not None:
                    metapath_exist = True
                    # x_j is already an output
                    cumulative_output.append(x_j)
                    triples = MetagraphHelper().get_triples(self.a_star[i][j])
                    for triple in triples:
                        # retain cooutputs
                        output = triple.cooutputs
                        if output is not None and output not in cumulative_output:
                            cumulative_output.append(output)
                        #... and edges
                        if isinstance(triple.edges, Edge):
                            edges = MetagraphHelper().get_edge_list([triple.edges])
                        else:
                            edges = MetagraphHelper().get_edge_list(triple.edges)

                        for edge in edges:
                            if edge not in cumulative_edges:
                                cumulative_edges.append(edge)

            if not metapath_exist:
                return None

            is_subset = True
            for elt in list(target):
                if elt not in cumulative_output:
                    is_subset = False
                    break
            if is_subset:
                for edge in cumulative_edges:
                    if edge not in metapaths:
                        metapaths.append(edge)

        valid_metapaths = []
        from itertools import combinations
        all_subsets = sum(map(lambda r: list(combinations(metapaths, r)), range(1, len(metapaths)+1)), [])
        for path in all_subsets:
            if len(path) <= len(metapaths):
                mp = Metapath(source, target, list(path))
                if self.is_metapath(mp):
                    valid_metapaths.append(mp)

        #print('valid metapath computation- %s'%(time.time()- start))
        return valid_metapaths

    def get_all_metapaths_from3(self, source, target):
        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])

        # check subset
        if not source.intersection(self.generating_set) == source:
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.intersection(self.generating_set) == target:
            raise MetagraphException('target', resources['not_a_subset'])

        # compute A* first
        if self.a_star is None:
            self.a_star = self.get_closure().tolist()

        metapaths = []
        all_applicable_input_rows = []
        for x_i in source:
            index = list(self.generating_set).index(x_i)
            if index not in all_applicable_input_rows:
                all_applicable_input_rows.append(index)


        cumulative_output_global = []
        cumulative_edges_global = []
        for i in all_applicable_input_rows:
            mp_exist_for_row=False
            cumulative_output_local = []
            cumulative_edges_local = []
            for x_j in target:
                j = list(self.generating_set).index(x_j)

                if self.a_star[i][j] is not None:
                    mp_exist_for_row = True

                    # x_j is already an output
                    cumulative_output_local.append(x_j)
                    triples = MetagraphHelper().get_triples(self.a_star[i][j])
                    for triple in triples:
                        # retain cooutputs
                        output = triple.cooutputs
                        if output is not None:
                            cumulative_output_local += output
                        if output is not None:
                            cumulative_output_global += output

                        #... and edges
                        if isinstance(triple.edges, Edge):
                            edges = MetagraphHelper().get_edge_list([triple.edges])
                        else:
                            edges = MetagraphHelper().get_edge_list(triple.edges)

                        for edge in edges:
                            if edge not in cumulative_edges_local:
                                cumulative_edges_local.append(edge)
                            if edge not in cumulative_edges_global:
                                cumulative_edges_global.append(edge)

            if not mp_exist_for_row:
               continue

            # check if cumulative outputs form a cover for the target
            if set(target).issubset(set(cumulative_output_local)):
                if set(cumulative_edges_local) not in metapaths:
                    metapaths.append(set(cumulative_edges_local))

            elif set(target).issubset(set(cumulative_edges_global)):
                if set(cumulative_edges_global) not in metapaths:
                    metapaths.append(set(cumulative_edges_global))

            else:
                break

        if len(metapaths)>0:
            result=[]
            for path in metapaths:
                # check no loops
                #if self.check_no_loops(list(path)):
                mp = Metapath(source, target, list(path))
                if self.is_metapath(mp):
                   result.append(mp)
            return result
        else:
            return None

    def is_metapath(self, metapath_candidate):
        """ Checks if the given candidate is a metapath.
        :param metapath_candidate: Metapath object
        :return: boolean
        """

        if metapath_candidate is None:
            raise MetagraphException('metapath_candidate', resources['value_null'])

        all_inputs=[]
        all_outputs=[]
        for edge in metapath_candidate.edge_list:
            for input_elt in list(edge.invertex):
                if input_elt not in all_inputs:
                    all_inputs.append(input_elt)
            for output_elt in list(edge.outvertex):
                if output_elt not in all_outputs:
                    all_outputs.append(output_elt)

        # now check input and output sets
        if (set(all_inputs).difference(set(all_outputs)).issubset(metapath_candidate.source)) and \
           set(metapath_candidate.target).issubset(all_outputs):
            return True

        return False

    def is_edge_dominant_metapath(self, metapath):
        """ Checks if the given metapath is an edge-dominant metapath.
        :param metapath: Metapath object
        :return: boolean
        """

        if metapath is None:
            raise MetagraphException('metapath', resources['value_null'])

        from itertools import combinations
        # check input metapath is valid
        if not self.is_metapath(metapath):
            return False

        all_subsets = sum(map(lambda r: list(combinations(metapath.edge_list, r)),
                              range(1, len(metapath.edge_list)+1)), [])
        # if one proper subset is a metapath then not edge dominant
        for path in all_subsets:
            # must be a proper subset
            if len(path) < len(metapath.edge_list):
                mp = Metapath(metapath.source, metapath.target, list(path))
                if self.is_metapath(mp):
                    return False

        return True

    def is_input_dominant_metapath(self, metapath):
        """ Checks if the given metapath is an input-dominant metapath.
        :param metapath: Metapath object
        :return: boolean
        """

        if metapath is None:
            raise MetagraphException('metapath', resources['value_null'])

        from itertools import combinations
        # check input metapath is valid
        if not self.is_metapath(metapath):
            return False

        # get all proper subsets of subset1
        all_subsets = sum(map(lambda r: list(combinations(metapath.source, r)), range(1, len(metapath.source)+1)), [])
        # if one proper subset has a metapath to subset2 then not input dominant
        for subset in all_subsets:
            # must be proper subset
            if len(subset) < len(metapath.source):
                if isinstance(subset, tuple):
                    subset = set(list(subset))
                metapath1 = self.get_all_metapaths_from(subset, metapath.target)
                if metapath1 is not None and len(metapath1) > 0:
                    #print('source: %s, target: %s'%(subset, metapath.target))
                    return False
        return True

    def is_dominant_metapath(self, metapath):
        """ Checks if the given metapath is a dominant metapath.
        :param metapath: Metapath object
        :return: boolean
        """

        if metapath is None:
            raise MetagraphException('metapath', resources['value_null'])

        # check input metapath is valid
        if not self.is_metapath(metapath):
            return False

        if (self.is_edge_dominant_metapath(metapath) and
           self.is_input_dominant_metapath(metapath)):

            return True

        return False

    def is_redundant_edge(self, edge, metapath, source, target):
        """ Checks if the given edge is redundant for the given metapath.
        :param edge: Edge object
        :param metapath: Metapath object
        :param source: set
        :param target: set
        :return: boolean
        """

        if edge is None:
            raise MetagraphException('edge', resources['value_null'])
        if metapath is None:
            raise MetagraphException('metapath', resources['value_null'])
        if source is None:
            raise MetagraphException('source', resources['value_null'])
        if target is None:
            raise MetagraphException('target', resources['value_null'])

        # check input metapath is valid
        if not self.is_metapath(metapath):
            raise MetagraphException('metapath', resources['arguments_invalid'])

        from itertools import combinations
        all_subsets = sum(map(lambda r: list(combinations(target, r)), range(1, len(target)+1)), [])
        # get all metapaths from subset1 to proper subsets of subset2
        for subset in all_subsets:
            if len(subset) < len(target):
                metapaths = self.get_all_metapaths_from(source, target)
                # check if edges is in every metapath found
                if metapaths is not None and len(metapaths) > 0:
                    for mp in metapaths:
                        if not MetagraphHelper().is_edge_in_list(edge, mp.edge_list):
                            # redundant
                            return True

                    # non redundant
                    return False
        return False

    def is_cutset_temp(self, edge_list, source, target):
        """ Checks if an edge list is a cutset between a given source and target.
        :param edge_list: list of Edge objects
        :param source: set
        :param target: set
        :return: boolean
        """

        if edge_list is None:
            raise MetagraphException('edges', resources['value_null'])
        if source is None:
            raise MetagraphException('source', resources['value_null'])
        if target is None:
            raise MetagraphException('target', resources['value_null'])

        # remove input edge list from original list
        original_edges = self.edges
        modified_edge_list = []
        for edge1 in original_edges:
            included = False
            for edge2 in edge_list:
                if edge1.invertex == edge2.invertex and edge1.outvertex == edge2.outvertex:
                    included = True
                    break
            if not included:
                modified_edge_list.append(edge1)

        mg = Metagraph(self.generating_set)
        mg.add_edges_from(modified_edge_list)
        #adjacency_matrix = mg.adjacency_matrix().tolist()

        metapaths = mg.get_all_metapaths_from(source, target)

        if metapaths is not None and len(metapaths) > 0:
            return False

        return True

    def is_cutset(self, edge_list, source, target):
        """ Checks if an edge list is a cutset between a given source and target.
        :param edge_list: list of Edge objects
        :param source: set
        :param target: set
        :return: boolean
        """

        if edge_list is None:
            raise MetagraphException('edges', resources['value_null'])
        if source is None:
            raise MetagraphException('source', resources['value_null'])
        if target is None:
            raise MetagraphException('target', resources['value_null'])

        # get all metapaths
        metapaths = self.get_all_metapaths_from(source, target)
        if metapaths is not None and len(metapaths)>0:
           for metapath in metapaths:
                updated = MetagraphHelper().remove_edge_list(edge_list, metapath.edge_list)
                mpc = Metapath(source, target, updated)
                if self.is_metapath(mpc):
                    # not a cutset
                    return False

           # is a cutset
           return True

        return False

    def get_minimal_cutset(self, source, target):
        """ Retrieves the minimal cutset between a given source and target.
        :param source: set
        :param target: set
        :return: list of Edge objects
        """

        if source is None:
            raise MetagraphException('source', resources['value_null'])
        if target is None:
            raise MetagraphException('target', resources['value_null'])

        metapaths = self.get_all_metapaths_from(source, target)
        if metapaths is None or len(metapaths) == 0:
            return None

        cutsets = []
        from itertools import combinations
        # noinspection PyTypeChecker
        for metapath in metapaths:
            all_combinations = sum(map(lambda r: list(combinations(metapath.edge_list, r)),
                                   range(1, len(metapath.edge_list)+1)), [])
            for combination in all_combinations:
                if self.is_cutset(list(combination), source, target) and list(combination) not in cutsets:
                    cutsets.append(list(combination))

        # return smallest cutset
        if len(cutsets) > 0:
            smallest = cutsets[0]
            for cutset in cutsets:
                if len(cutset) < len(smallest):
                    smallest = cutset
            return smallest

        return None

    def is_bridge(self, edge_list, source, target):
        """ Checks if a given edge list forms a bridge between a source and a target.
        :param edge_list: list of Edge objects
        :param source: set
        :param target: set
        :return: boolean
        """
        if edge_list is None:
            raise MetagraphException('edge_list', resources['value_null'])
        if source is None:
            raise MetagraphException('source', resources['value_null'])
        if target is None:
            raise MetagraphException('target', resources['value_null'])
        if len(edge_list)!=1:
            return False

        if not isinstance(edge_list, list):
            raise MetagraphException('edge_list', resources['arguments_invalid'])

        return self.is_cutset(edge_list, source, target)

    def get_projection(self, generator_subset):
        """ Gets the metagraph projection for a subset of the generating set.
        :param generator_subset: set
        :return: Metagraph object
        """

        # gen set size N
        # edge count C
        # proj elts M

        if generator_subset is None:
            raise MetagraphException('generator_subset', resources['value_null'])

        # step1. reduce A* by removing unwanted rows, cols
        applicable_rows_and_cols = []
        for x_i in generator_subset:
            index = list(self.generating_set).index(x_i)
            if index not in applicable_rows_and_cols:
                applicable_rows_and_cols.append(index)

        a_star = self.get_closure().tolist()

        # sort list
        applicable_rows_and_cols = sorted(applicable_rows_and_cols)

        size = len(generator_subset)
        a_star_new = MetagraphHelper().get_null_matrix(size, size)

        # O(M2)
        m = 0
        for i in applicable_rows_and_cols:
            n = 0
            for j in applicable_rows_and_cols:
                a_star_new[m][n] = a_star[i][j]
                n += 1
            m += 1

        # step2. create list L from edges in E (not E') s.t. V_e is a subset of X'
        edge_list1 = []
        all_triples = []
        k = len(applicable_rows_and_cols)
        # O(M2C2)
        for i in range(k):
            for j in range(k):
                if a_star_new[i][j] is not None:
                    triples = MetagraphHelper().get_triples(a_star_new[i][j])
                    for triple in triples:
                        if isinstance(triple.edges, Edge):
                            new_triple = Triple(triple.coinputs, triple.cooutputs, [triple.edges])
                            if new_triple not in all_triples:
                                all_triples.append(new_triple)
                            edges = MetagraphHelper().extract_edge_list([triple.edges])
                        else:
                            if triple not in all_triples:
                                all_triples.append(triple)
                            edges = MetagraphHelper().extract_edge_list(triple.edges)

                        # select edges with invertices in generator_subset
                        for edge in edges:
                            if edge.invertex.issubset(set(generator_subset)) and ([edge] not in edge_list1):
                                edge_list1.append([edge])

        # step3. find combinations of triples s.t. union(CI_t_i)\ union(CO_t_i) is a subset of generator_subset
        from itertools import combinations
        #all_combs=nCr=n!/r!(n-r)! = C, C(C-1)/2, C(C-1)(C-2)/3 to 1  (r=1 to C)
        all_combinations = sum(map(lambda r: list(combinations(all_triples, r)), range(1, len(all_triples)+1)), [])
        # O(NC!))
        for combination in all_combinations:
            # O(NC)
            coinput = MetagraphHelper().get_coinputs_from_triples(combination)
            cooutput = MetagraphHelper().get_cooutputs_from_triples(combination)
            diff = set(coinput).difference(set(cooutput))
            if diff.issubset(set(generator_subset)):
                # add edges in combination to L
                edges2 = MetagraphHelper().get_edges_from_triple_list(list(combination))
                included = MetagraphHelper().is_edge_list_included_recursive(edges2, edge_list1)
                if not included:
                    edge_list1.append(edges2)

        # step4. construct L0 from L
        triples_list_l0 = []
        # O(C2N)
        for element in edge_list1:
            all_inputs = MetagraphHelper().get_netinputs(element)
            all_outputs = MetagraphHelper().get_netoutputs(element)
            net_inputs = list(set(all_inputs).difference(all_outputs))
            net_outputs = all_outputs
            new_triple = Triple(set(net_inputs), set(net_outputs), element)
            if new_triple not in triples_list_l0:
                triples_list_l0.append(new_triple)

        # step5. reduce L0
        to_eliminate = []
        # O(C4)
        for i in triples_list_l0:
            for j in triples_list_l0:
                if i != j:
                    # check if i is subsumed by j
                    edges_i = MetagraphHelper().get_edge_list(i.edges)
                    edges_j = MetagraphHelper().get_edge_list(j.edges)
                    outputs_i = i.cooutputs
                    outputs_j = j.coinputs

                    # check j's edges are a subset of i's edges
                    subset = True
                    for edge in edges_j:
                        if edge not in edges_i:
                            subset = False
                            break

                    inclusive = True
                    if subset:
                        # edges form a subset..check outputs are inclusive
                        outputs_j_in_x = []
                        for output in outputs_j:
                            if (output in generator_subset) and (output not in outputs_j_in_x):
                                outputs_j_in_x.append(output)

                        outputs_i_in_x = []
                        for output in outputs_i:
                            if (output in generator_subset) and (output not in outputs_i_in_x):
                                outputs_i_in_x.append(output)

                        for output in outputs_i_in_x:
                            if output not in outputs_j_in_x:
                                inclusive = False
                                break

                        if inclusive and (i not in to_eliminate):
                            to_eliminate.append(i)
                            break

        for elt in to_eliminate:
            triples_list_l0.remove(elt)

        to_drop = []

        for i in triples_list_l0:
            for j in triples_list_l0:
                if i != j:
                    inputs_i = i.coinputs
                    inputs_j = j.coinputs
                    outputs_i = i.cooutputs
                    outputs_j = j.cooutputs

                    # check if input and output of j are a subset of i
                    input_subset = True
                    for input_elt in inputs_j:
                        if input_elt not in inputs_i:
                            input_subset = False
                            break

                    if input_subset:
                        output_subset = True
                        for output in outputs_j:
                            if output not in outputs_i:
                                output_subset = False
                                break

                        if output_subset:
                            for elt in j.cooutputs:
                                i.cooutputs.remove(elt)
                            if (i.cooutputs is None or len(i.cooutputs) == 0) and (i not in to_drop):
                                to_drop.append(i)

        for item in to_drop:
            triples_list_l0.remove(item)

        #step6. merge triples based on identical inputs and outputs
        triples_to_merge = dict()
        index = 0
        for triple1 in triples_list_l0:
            for triple2 in triples_list_l0:
                if triple1 != triple2:
                    # merge if same input and output
                    if triple1.coinputs == triple2.coinputs and triple1.cooutputs == triple2.cooutputs:
                        if index not in triples_to_merge:
                            triples_to_merge[index] = []
                        if triple2 not in triples_to_merge[index]:
                            triples_to_merge[index].append(triple2)
            index += 1

        post_merge_triples = dict()
        for index, triples_list in triples_to_merge.iteritems():
            triple1 = triples_to_merge[index]
            if triple1 in triples_list_l0:
                triples_list_l0.remove(triple1)
                for triple2 in triples_list:
                    if triple2 in triples_list_l0:
                        triples_list_l0.remove(triple2)
                    if index not in post_merge_triples:
                        post_merge_triples[index] = Triple(triple1.coinputs, triple1.cooutputs,
                                                           [triple1.edges, triple2.edges])
                    else:
                        post_merge_triples[index].edges.append(triple2.edges)
                triples_list_l0.append(post_merge_triples[index])

        #step7. and triples with identical inputs only
        triples_to_merge = dict()
        index = 0
        for triple1 in triples_list_l0:
            for triple2 in triples_list_l0:
                if triple1 != triple2:
                    # merge if same input
                    if triple1.coinputs == triple2.coinputs:
                        if index not in triples_to_merge:
                            triples_to_merge[index] = []
                        if triple2 not in triples_to_merge[index]:
                            triples_to_merge[index].append(triple2)
            index += 1

        triple_list_l0_copy = copy.copy(triples_list_l0)

        post_merge_triples = dict()
        #O(C3)
        for index, triples_list in triples_to_merge.iteritems():
            triple1 = triple_list_l0_copy[index]
            if triple1 in triples_list_l0:
                triples_list_l0.remove(triple1)
                for triple2 in triples_list:
                    if triple2 in triples_list_l0:
                        triples_list_l0.remove(triple2)
                    if index not in post_merge_triples:
                        edges_to_merge = list(triple1.edges)
                        for elt in list(triple2.edges):
                            if elt not in edges_to_merge:
                                edges_to_merge.append(elt)
                        post_merge_triples[index] = Triple(triple1.coinputs, triple1.cooutputs.union(triple2.cooutputs),
                                                           edges_to_merge)
                    else:
                        edges_to_merge = list(post_merge_triples[index].edges)
                        for elt in list(triple2.edges):
                            if elt not in edges_to_merge:
                                edges_to_merge.append(elt)
                        post_merge_triples[index] = Triple(post_merge_triples[index].coinputs,
                                                           post_merge_triples[index].cooutputs.union(triple2.cooutputs),
                                                           edges_to_merge)  # triple1
                triples_list_l0.append(post_merge_triples[index])

        temp_list = []
        for triple in triples_list_l0:
            # remove all inputs and outputs that are not in generator_subset
            valid_inputs = triple.coinputs.intersection(set(generator_subset))
            valid_outputs = triple.cooutputs.intersection(set(generator_subset))
            new_triple = Triple(valid_inputs, valid_outputs, triple.edges)
            if new_triple not in temp_list:
                temp_list.append(new_triple)

        # remove any tuples with zero input or output
        final_triples_list = []
        for triple in temp_list:
            if triple.coinputs is not None and triple.cooutputs is not None and \
               len(triple.coinputs) > 0 and len(triple.cooutputs) > 0 and \
               (triple not in final_triples_list):
                final_triples_list.append(triple)

        edge_list = []
        for triple in final_triples_list:
            edge = Edge(triple.coinputs, triple.cooutputs, None, repr(triple.edges))
            if edge not in edge_list:
                edge_list.append(edge)

        if edge_list is None or len(edge_list) == 0:
            return None

        mg = Metagraph(set(generator_subset))
        mg.add_edges_from(edge_list)

        return mg

    def incidence_matrix(self):
        """ Gets the metagraph projection for a subset of the generating set.
        :return: numpy.matrix
        """

        rows = len(self.generating_set)
        cols = len(self.edges)

        incidence_matrix = MetagraphHelper().get_null_matrix(rows, cols)

        for i in range(rows):
            x_i = list(self.generating_set)[i]
            for j in range(cols):
                e_j = self.edges[j]
                if x_i in e_j.invertex:
                    incidence_matrix[i][j] = -1
                elif x_i in e_j.outvertex:
                    incidence_matrix[i][j] = 1
                else:
                    incidence_matrix[i][j] = None

        # noinspection PyCallingNonCallable
        return matrix(incidence_matrix)

    def get_inverse(self):
        """ Gets the inverse metagraph.
        :return: Metagraph object
        """

        # gen set size N
        # edge count C
        # proj elts M

        incidence_m = self.incidence_matrix().tolist()

        edge_list = []
        # step1: extract indices
        col_index = 0
        # O(N3)
        for i in range(len(incidence_m[0])):
            negative_item_indices = []
            positive_item_indices = []
            column = [row[col_index] for row in incidence_m]
            row_index = 0
            for item1 in column:
                if item1 == -1:
                   # get all elements with +1 across the row
                    row = incidence_m[row_index]
                    eligible = []
                    local_index = 0
                    for item2 in row:
                        if item2 == 1 and local_index not in eligible:
                            eligible.append(local_index)
                        local_index += 1

                    if len(eligible) == 0:
                        row_index += 1  # debug dr check this
                        continue

                    # TODO: how do we handle multiple occurrences of +1?
                    # keep a track of -1 and + 1 indices
                    if (row_index, col_index) not in negative_item_indices:
                        negative_item_indices.append((row_index, col_index))

                    for local_index in eligible:
                        if (row_index, local_index) not in positive_item_indices:
                            positive_item_indices.append((row_index, local_index))

                row_index += 1

            # construct edges from indices
            invertex = []
            outvertex = []
            edge_label = None
            for negative_item_index in negative_item_indices:
                if repr(self.edges[negative_item_index[1]]) not in outvertex:
                    outvertex.append(repr(self.edges[negative_item_index[1]]))

            for positive_item_index in positive_item_indices:
                if repr(self.edges[positive_item_index[1]]) not in invertex:
                    invertex.append(repr(self.edges[positive_item_index[1]]))

                # generate label
                if edge_label is None:
                    edge_label = '<%s,%s>' % (list(self.generating_set)[positive_item_index[0]],
                                              repr(self.edges[positive_item_index[1]]))
                else:
                    edge_label += ', <%s,%s>' % (list(self.generating_set)[positive_item_index[0]],
                                                 repr(self.edges[positive_item_index[1]]))

            if invertex is not None and outvertex is not None and len(invertex) > 0 and len(outvertex) > 0:
                edge = Edge(set(invertex), set(outvertex), None, edge_label)
                if edge not in edge_list:
                    edge_list.append(edge)
            col_index += 1

        # compress the edges
        compressed_edges = []
        for edge1 in edge_list:
            compressed = False
            for edge2 in edge_list:
                if edge1 != edge2:
                    if edge1.invertex == edge2.invertex and edge1.label == edge2.label:
                        new_edge = Edge(edge1.invertex, edge1.outvertex.union(edge2.outvertex), None, edge1.label)
                        if not MetagraphHelper().is_edge_in_list(new_edge, compressed_edges):
                            compressed_edges.append(new_edge)
                        compressed = True
            if not compressed and (not MetagraphHelper().is_edge_in_list(edge1, compressed_edges)):
                compressed_edges.append(edge1)

        # add links to alpha and beta
        row_index = 0
        occurrences = lambda s, lst: (y for y, e in enumerate(lst) if e == s)
        for row in incidence_m:
            if row.__contains__(-1) and (not row.__contains__(1)):
                col_indices = list(occurrences(-1, row))
                for col_index in col_indices:
                    label = '<%s, alpha>' % (list(self.generating_set)[row_index])
                    new_edge = Edge({'alpha'}, {repr(self.edges[col_index])}, None, label)
                    if not MetagraphHelper().is_edge_in_list(new_edge, compressed_edges):
                        compressed_edges.append(new_edge)

            elif row.__contains__(1) and (not row.__contains__(-1)):
                col_indices = list(occurrences(1, row))
                for col_index in col_indices:
                    label = '<%s, %s>' % (list(self.generating_set)[row_index], repr(self.edges[col_index]))
                    new_edge = Edge({repr(self.edges[col_index])}, {'beta'}, None, label)
                    if not MetagraphHelper().is_edge_in_list(new_edge, compressed_edges):
                        compressed_edges.append(new_edge)
            row_index += 1

        mg = Metagraph(MetagraphHelper().get_generating_set(compressed_edges))
        mg.add_edges_from(compressed_edges)
        return mg

    def get_efm(self, generator_subset):
        """ Gets the element-flow metagraph.
        :param generator_subset: set
        :return: Metagraph object
        """

        # gen set size N
        # edge count C
        # efm elts M

        if generator_subset is None or len(generator_subset) == 0:
            raise MetagraphException('generator_subset', resources['value_null'])

        incidence_m = self.incidence_matrix().tolist()
        excluded = self.generating_set.difference(generator_subset)

        # compute G1 and G2
        applicable_rows = []
        # O(M)
        for x_i in generator_subset:
            index = list(self.generating_set).index(x_i)
            if index not in applicable_rows:
                applicable_rows.append(index)

        # sort list
        applicable_rows = sorted(applicable_rows)
        inapplicable_rows = sorted(set(range(len(self.generating_set))).difference(applicable_rows))

        rows1 = len(generator_subset)
        cols1 = len(incidence_m[0])
        rows2 = len(excluded)
        g1 = MetagraphHelper().get_null_matrix(rows1, cols1)
        g2 = MetagraphHelper().get_null_matrix(rows2, cols1)

        m = 0
        for i in applicable_rows:
            n = 0
            for j in range(cols1):
                g1[m][n] = incidence_m[i][j]
                n += 1
            m += 1

        m = 0
        for i in inapplicable_rows:
            n = 0
            for j in range(cols1):
                g2[m][n] = incidence_m[i][j]
                n += 1
            m += 1

        g1_t = MetagraphHelper().transpose_matrix(g1)
        # O(N3)
        mult_r = MetagraphHelper().custom_multiply_matrices(g2, g1_t, self.edges)
        row_index = 0
        edge_list = []
        lookup = dict()
        # O(N3)
        for row in mult_r:
            col_index = 0
            invertices = []
            outvertices = []
            # O(N2)
            for elt in row:
                if len(elt) > 0 and isinstance(list(elt)[0], tuple):
                    extracted = list(elt)[0]
                    if 1 in extracted:
                        invertex = []
                        local_indices = [row.index(elt2) for elt2 in row if elt.issubset(elt2)]
                        #O(N)
                        for local_index in local_indices:
                            value = list(self.generating_set)[applicable_rows[local_index]]
                            if value not in invertex:
                                invertex.append(value)

                        if set(invertex) not in invertices:
                            invertices.append(set(invertex))
                        if repr(invertex) not in lookup:
                            lookup[repr(invertex)] = extracted[1]

                    elif -1 in extracted:
                        outvertex = []
                        local_indices = [row.index(elt2) for elt2 in row if elt.issubset(elt2)]
                        for local_index in local_indices:
                            value = list(self.generating_set)[applicable_rows[local_index]]
                            if value not in outvertex:
                                outvertex.append(value)

                        if set(outvertex) not in outvertices:
                            outvertices.append(set(outvertex))
                        if repr(outvertex) not in lookup:
                            lookup[repr(outvertex)] = extracted[1]

                col_index += 1

            # combine the invertices and outvertices
            for invertex in invertices:
                for outvertex in outvertices:
                    # create flow composition
                    label = '%s <%s; %s>' % (list(self.generating_set)[inapplicable_rows[row_index]],
                                             lookup[repr(list(invertex))], lookup[repr(list(outvertex))])
                    edge = Edge(invertex, outvertex, None, label)
                    if not MetagraphHelper().is_edge_in_list(edge, edge_list):
                        edge_list.append(edge)

            row_index += 1

        # combine edges
        final_edge_list = []
        for edge1 in edge_list:
            match = False
            for edge2 in edge_list:
                if edge1 != edge2:
                    if edge1.invertex == edge2.invertex and edge1.outvertex == edge1.outvertex:
                        match = True
                        comp1 = MetagraphHelper().extract_edge_label_components(edge1.label)
                        comp2 = MetagraphHelper().extract_edge_label_components(edge2.label)
                        combined = (comp1[0].union(comp2[0]), comp1[1].union(comp2[1]), comp1[2].union(comp2[2]))
                        label = '%s <%s; %s>' % (list(combined[0]), list(combined[1]), list(combined[2]))
                        combined_edge = Edge(edge1.invertex, edge1.outvertex, None, label)
                        if not MetagraphHelper().is_edge_in_list(combined_edge, final_edge_list):
                            final_edge_list.append(combined_edge)
            if not match:
                if not MetagraphHelper().is_edge_in_list(edge1, final_edge_list):
                    final_edge_list.append(edge1)

        if len(final_edge_list) > 0:
            gen_set = MetagraphHelper().get_generating_set(final_edge_list)
            mg = Metagraph(gen_set)
            mg.add_edges_from(final_edge_list)
            return mg

        return None

    def dominates(self, metagraph2):
        """Checks if the metagraph dominates that provided.
        :param metagraph2: Metagraph object
        :return: boolean
        """

        if metagraph2 is None:
            raise MetagraphException('metagraph2', resources['value_null'])

        #adjacency_matrix = self.adjacency_matrix().tolist()
        #all_metapaths1 = []

        from itertools import combinations
        all_sources1 = sum(map(lambda r: list(combinations(self.generating_set, r)),
                               range(1, len(self.generating_set))), [])
        all_targets1 = copy.copy(all_sources1)

        all_sources2 = sum(map(lambda r: list(combinations(metagraph2.generating_set, r)),
                               range(1, len(metagraph2.generating_set))), [])
        all_targets2 = copy.copy(all_sources2)

        all_metapaths1 = []
        i = 1
        #s1 =  len(all_sources1)
        #t1 =  len(all_targets1)
        #s2 =  len(all_sources2)
        #t2 =  len(all_targets2)

        for source in all_sources1:
            for target in all_targets1:
                if source != target:
                    mp = self.get_all_metapaths_from(set(source), set(target))
                    if mp is not None and len(mp) > 0 and (mp not in all_metapaths1):
                        all_metapaths1.append(mp)
                    #print(i)
                    i += 1

        all_metapaths2 = []
        for source in all_sources2:
            for target in all_targets2:
                if source != target:
                    mp = self.get_all_metapaths_from(set(source), set(target))
                    if mp is not None and len(mp) > 0 and (mp not in all_metapaths2):
                        all_metapaths2.append(mp)

        for mp1 in all_metapaths2:
            dominated = False
            for mp2 in all_metapaths1:
                if mp2.dominates(mp1):
                    dominated = True
                    break
            if not dominated:
                return False

        return True

    def __repr__(self):
        edge_desc = [repr(edge) for edge in self.edges]
        full_desc = ''
        for desc in edge_desc:
            if full_desc == '':
                full_desc = desc
            else:
                full_desc += ", " + desc
        desc = '%s(%s)' % (str(type(self)), full_desc)
        desc = desc.replace('\\', '')
        return desc

class ConditionalMetagraph(Metagraph):
    """ Represents a conditional metagraph that is instantiated using a set of variables and a set of propositions.
    """

    def __init__(self, variables_set, propositions_set):
        if not isinstance(variables_set, set):
            raise MetagraphException('variable_set', resources['format_invalid'])
        if not isinstance(propositions_set, set):
            raise MetagraphException('propositions_set', resources['format_invalid'])
        if len(variables_set.intersection(propositions_set)) > 0:
            raise MetagraphException('variables_and_propositions', resources['partition_invalid'])
        self.nodes = []
        self.edges = []
        self.mg = None
        self.variables_set = variables_set
        self.propositions_set = propositions_set
        self.generating_set = variables_set.union(propositions_set)
        super(ConditionalMetagraph, self).__init__(self.generating_set)

    def add_edges_from(self, edge_list):
        """ Adds the given list of edges to the conditional metagraph.
        :param edge_list: list of Edge objects
        :return: None
        """
        if edge_list is None or len(edge_list) == 0:
            raise MetagraphException('edge_list', resources['value_null'])

        for edge in edge_list:
            if not isinstance(edge, Edge):
                raise MetagraphException('edge', resources['format_invalid'])
            if len(edge.invertex.union(edge.outvertex)) == 0:
                raise MetagraphException('edge', resources['value_null'])
             # if outvertex contains a proposition, the outvertex cannot contain any other element
            for proposition in self.propositions_set:
                if proposition in edge.outvertex:
                    if len(edge.outvertex) > 1:
                        raise MetagraphException('edge', resources['value_invalid'])

        for edge in edge_list:
            node1 = Node(edge.invertex)
            node2 = Node(edge.outvertex)
            if not MetagraphHelper().is_node_in_list(node1, self.nodes):
                self.nodes.append(node1)
            if not MetagraphHelper().is_node_in_list(node2, self.nodes):
                self.nodes.append(node2)
            if edge not in self.edges: #debug dr dr
               self.edges.append(edge)

        self.edges = edge_list

    def add_metagraph(self, metagraph2):
        """ Adds the given conditional metagraph to current and returns the composed result.
        :param metagraph2: ConditonalMetagraph object
        :return: ConditonalMetagraph object
        """

        if metagraph2 is None:
            raise MetagraphException('metagraph2', resources['value_null'])

        generating_set1 = self.generating_set
        generating_set2 = metagraph2.generating_set

        if generating_set2 is None or len(generating_set2) == 0:
            raise MetagraphException('metagraph2.generating_set', resources['value_null'])

        # check if the generating sets of the matrices overlap (otherwise no sense in combining metagraphs)
        #intersection=generating_set1.intersection(generating_set2)
        #if intersection==None:
        #    raise MetagraphException('generating_sets', resources['no_overlap'])

        if len(generating_set1.difference(generating_set2)) == 0 and \
           len(generating_set2.difference(generating_set1)) == 0:
            # generating sets are identical..simply add edges
            # size = len(generating_set1)
            self.variables_set = self.variables_set.union(metagraph2.variables_set)
            self.propositions_set = self.variables_set.union(metagraph2.propositions_set)
            for edge in metagraph2.edges:
                if edge not in self.edges:
                    self.add_edge(edge)
        else:
            # generating sets overlap but are different...combine generating sets and then add edges
            # combined_generating_set = generating_set1.union(generating_set2)
            self.generating_set = generating_set1.union(generating_set2)
            self.variables_set = self.variables_set.union(metagraph2.variables_set)
            self.propositions_set = self.propositions_set.union(metagraph2.propositions_set)
            for edge in metagraph2.edges:
                if edge not in self.edges:
                    self.add_edge(edge)

        return self

    def get_context(self, true_propositions, false_propositions):
        """Retrieves the context metagraph for the given true and false propositions.
        :param true_propositions: set
        :param false_propositions: set
        :return: ConditionalMetagraph object
        """
        if true_propositions is None or len(true_propositions) == 0:
            raise MetagraphException('true_propositions', resources['value_null'])
        if false_propositions is None or len(false_propositions) == 0:
            raise MetagraphException('false_propositions', resources['value_null'])

        for proposition in true_propositions:
            if proposition not in self.propositions_set:
                raise MetagraphException('true_propositions', resources['range_invalid'])
        for proposition in false_propositions:
            if proposition not in self.propositions_set:
                raise MetagraphException('false_propositions', resources['range_invalid'])

        edges_to_remove = []
        edges_copy = copy.copy(self.edges)
        for edge in edges_copy:
            for proposition in list(true_propositions):
                if proposition in edge.invertex:
                    edge.invertex.difference({proposition})
                    # remove if this results in an invertex that is null
                    if len(edge.invertex) == 0 and edge not in edges_to_remove:
                        edges_to_remove.append(edge)
                if proposition in edge.outvertex:
                    edge.outvertex.difference({proposition})
                    # remove if this results in an outvertex that is null
                    if len(edge.outvertex) == 0 and edge not in edges_to_remove:
                        edges_to_remove.append(edge)

            for proposition in list(false_propositions):
                if proposition in edge.invertex or proposition in edge.outvertex:
                    # remove edge
                    if edge not in edges_to_remove:
                        edges_to_remove.append(edge)

        # create new conditional metagraph describing context
        context = ConditionalMetagraph(self.variables_set, self.propositions_set)
        for edge in edges_to_remove:
            if edge in edges_copy:
                edges_copy.remove(edge)
        context.add_edges_from(edges_copy)

        return context

    def get_projection(self, variables_subset):
        """ Gets the conditional metagraph projection for a subset of its variable set.
        :param variables_subset: set
        :return: Metagraph object
        """
        if variables_subset is None or len(variables_subset) == 0:
            raise MetagraphException('variables_subset', resources['value_null'])

        subset = variables_subset.union(self.propositions_set)
        generator_set = self.variables_set.union(self.propositions_set)
        mg = Metagraph(generator_set)
        mg.add_edges_from(self.edges)
        return mg.get_projection(subset)

    def get_all_metapaths_from_old(self, source, target, prop_subset=None):
        """ Retrieves all metapaths between given source and target in the conditional metagraph.
        :param source: set
        :param target: set
        :return: list of Metapath objects
        """

        if source is None or len(source) == 0:
            raise MetagraphException('subset1', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('subset2', resources['value_null'])
        if not source.issubset(self.generating_set):
            raise MetagraphException('subset1', resources['not_a_subset'])
        if not target.issubset(self.generating_set):
            raise MetagraphException('subset2', resources['not_a_subset'])

        generator_set = self.variables_set.union(self.propositions_set)
        if self.mg is None:
            self.mg = Metagraph(generator_set)
            self.mg.add_edges_from(self.edges)
        if prop_subset is not None:
            return self.mg.get_all_metapaths_from(source.union(prop_subset), target)
        else:
            return self.mg.get_all_metapaths_from(source.union(self.propositions_set), target)

    def get_all_metapaths_from(self, source, target, include_propositions=False):
        if source==None or len(source)==0:
            raise MetagraphException('subset1', resources['value_null'])
        if target==None or len(target)==0:
            raise MetagraphException('subset2', resources['value_null'])
        if not source.issubset(self.generating_set):
            raise MetagraphException('subset1', resources['not_a_subset'])
        if not target.issubset(self.generating_set):
            raise MetagraphException('subset2', resources['not_a_subset'])

        generator_set=self.variables_set.union(self.propositions_set)
        if self.mg==None:
            self.mg=Metagraph(generator_set)
            self.mg.add_edges_from(self.edges)

        if include_propositions:
            return self.mg.get_all_metapaths_from(source.union(self.propositions_set), target)

        return self.mg.get_all_metapaths_from(source, target)

    # debug dr
    def get_all_metapaths(self):
        """ Retrieves all metapaths in the conditional metagraph.
        :return: List of Metapath objects
        """

        from itertools import combinations
        all_subsets=sum(map(lambda r: list(combinations(self.nodes, r)), range(1, len(self.nodes)+1)), [])
        #all_subsets = self.nodes

        cap_reached = False
        all_metapaths = []
        for subset1 in all_subsets:
            for subset2 in all_subsets:
                if subset1 != subset2:

                    if len(subset1)>1:
                        element_set=set()
                        for node in subset1:
                            element_set = element_set.union(node.element_set)
                        node1 = Node(element_set)
                    else:
                        node1=subset1[0]

                    if len(subset2)>1:
                        element_set=set()
                        for node in subset2:
                            element_set = element_set.union(node.element_set)
                        node2 = Node(element_set)
                    else:
                        node2=subset2[0]

                    # TODO: can source and target in a metapath overlap?
                    if not MetagraphHelper().nodes_overlap([node1], [node2]):
                        source = MetagraphHelper().get_element_set([node1])
                        target = MetagraphHelper().get_element_set([node2])
                        mps = self.get_all_metapaths_from(source, target)
                        if mps is None or len(mps) == 0:
                            continue
                        # noinspection PyTypeChecker
                        for mp in mps:
                            if mp not in all_metapaths:
                                all_metapaths.append(mp)
                        #if len(all_metapaths) >= 10:
                        #    cap_reached = True
                        #    break

            #if cap_reached:
            #    break

        return all_metapaths

    def has_conflicts1(self, metapath):
        """Checks whether the given metapath has any conflicts.
        :param metapath: Metapath object
        :return: boolean
        """

        invertices = set()
        intersec = set()
        edges = metapath.edge_list
        for edge in edges:
            invertices = invertices.union(edge.invertex)
            if len(intersec)==0:
                intersec = set(edge.attributes)
            else:
                intersec = intersec.intersection(set(edge.attributes))

        potential_conflicts_set = invertices.intersection(self.propositions_set)
        if self.edge_attributes_conflict(potential_conflicts_set, intersec):
            return True

        return False

    def has_conflicts_1(self, metapath, conflict_sources=[]):
        edges=metapath.edge_list
        filtered=[]

        for edge in edges:
            # check if edge should be discarded
            inv = edge.invertex.difference(edge.attributes)
            inv_int = inv.intersection(metapath.source)
            outv_int = edge.outvertex.intersection(metapath.target)
            if (inv_int is None or len(inv_int)==0) or \
               (outv_int is None or len(outv_int)==0):
                # discard edge
                continue

            if edge not in filtered:
                filtered.append(edge)

        invertices=set()
        intersec=set()
        for edge1 in filtered:
            for edge2 in filtered:
                if edge1!=edge2:
                    invertices= edge1.invertex.union(edge2.invertex)
                    intersec = set(edge1.attributes).intersection(set(edge2.attributes))
                    potential_conflicts_set = invertices.intersection(self.propositions_set)
                    if self.edge_attributes_conflict(potential_conflicts_set, intersec, conflict_sources):
                        return True

        return False

    def has_conflicts(self, metapath, conflict_sources=[], conflicting_edges=[]):
        edges=metapath.edge_list
        filtered=[]
        conflicts=False

        for edge in edges:
            # check if edge should be discarded
            inv = edge.invertex.difference(edge.attributes)
            inv_int = inv.intersection(metapath.source)
            outv_int = edge.outvertex.intersection(metapath.target)
            if (inv_int is None or len(inv_int)==0) or \
               (outv_int is None or len(outv_int)==0):
                # discard edge
                continue

            if edge not in filtered:
                filtered.append(edge)

        invertices=set()
        intersec=set()
        for edge1 in filtered:
            for edge2 in filtered:
                if edge1!=edge2:
                    invertices= edge1.invertex.union(edge2.invertex)
                    intersec = set(edge1.attributes).intersection(set(edge2.attributes))
                    potential_conflicts_set = invertices.intersection(self.propositions_set)
                    conflict_sources=[]
                    if self.edge_attributes_conflict(potential_conflicts_set, intersec, conflict_sources):
                        conflicting_edges.append((edge1,edge2, conflict_sources))
                        conflicts= True

        return conflicts

    def has_redundancies2(self, metapath, redundant_edges=[]):
        edges=metapath.edge_list
        filtered=[]
        redundancies=False

        for edge in edges:
            # check if edge should be discarded
            inv = edge.invertex.difference(edge.attributes)
            inv_int = inv.intersection(metapath.source)
            outv_int = edge.outvertex.intersection(metapath.target)
            if (inv_int is None or len(inv_int)==0) or \
               (outv_int is None or len(outv_int)==0):
                # discard edge
                continue

            if edge not in filtered:
                filtered.append(edge)

        for edge1 in filtered:
            for edge2 in filtered:
                if edge1!=edge2:
                    invertices= edge1.invertex.union(edge2.invertex)
                    prop_int = invertices.intersection(self.propositions_set)
                    #print('check edge overlap: %s, %s'%(edge1,edge2))
                    if self.edge_attributes_overlap(prop_int):
                        #print('actual redundancy')
                        redundant_edges.append((edge1,edge2))
                        redundancies= True

        return redundancies


    def has_redundancies(self, metapath):
        """ Checks if given metapath has redundancies.
        :param metapath: Metapath object
        :return: boolean
        """

        # check input metapath is valid
        if not self.is_metapath(metapath):
            raise MetagraphException('metapath', resources['arguments_invalid'])

        return not self.is_dominant_metapath(metapath)

    def edge_attributes_conflict1(self, potential_conflicts_set, intersecting_attr_set):
        """ Checks if given edge attributes conflict.
        :param potential_conflicts_set: set
        :return: boolean
        """

        if potential_conflicts_set is None:
            raise MetagraphException('potential_conflicts_set', resources['value_null'])

        # currently checks if actions conflict
        # extend later to include active times etc
        actions = self.get_actions(potential_conflicts_set)

        malware_sigs = self.get_malware_sigs(potential_conflicts_set)
        sig_present = self.get_sig_present(potential_conflicts_set)

        # TODO: extend to support more attributes
        if len(intersecting_attr_set)>0:
            # check if diff response actions are applied to same malware sig list
            if len(actions)>1 and len(malware_sigs)>1 and len(sig_present)==1:
                intersection = set(malware_sigs[0])
                for sig_list in malware_sigs:
                    intersection = intersection.intersection(set(sig_list))
                if len(intersection)>0:
                    return True

            # check if same response action is applied to different sig lists
            if len(malware_sigs)>1 and len(actions)==1 and len(sig_present)==1:
                return True

        if ('allow' in actions or 'permit' in actions) and 'deny' in actions: # len(intersecting_attr_set)>0 and
            return True

        return False

    def edge_attributes_conflict(self, potential_conflicts_set, intersecting_attr_set, conflict_sources):
        if potential_conflicts_set==None:
           raise MetagraphException('potential_conflicts_set',resources['value_null'])

        # currently checks if actions conflict
        # extend later to include active times etc
        actions= self.get_actions(potential_conflicts_set)
        protocols = self.get_protocols(potential_conflicts_set)
        tcp_dports = self.get_tcp_ports(potential_conflicts_set,True)
        udp_dports = self.get_udp_ports(potential_conflicts_set,True)
        tcp_sports = self.get_tcp_ports(potential_conflicts_set,False)
        udp_sports = self.get_udp_ports(potential_conflicts_set,False)

        priorities= self.get_priorities(potential_conflicts_set)
        interfaces= self.get_interfaces(potential_conflicts_set)
        middlebox1_devices= self.get_middleboxes(potential_conflicts_set,1)
        middlebox2_devices= self.get_middleboxes(potential_conflicts_set,2)
        malware_profiles= self.get_malware_profiles(potential_conflicts_set)
        url_filter_profiles= self.get_url_filter_profiles(potential_conflicts_set)
        spyware_profiles= self.get_spyware_profiles(potential_conflicts_set)

        #spyware_sev_lists = self.get_spyware_severity_lists(potential_conflicts_set)
        spyware_alert_lists= self.get_url_filter_lists('spyware_alert_list=', potential_conflicts_set)
        spyware_block_lists= self.get_url_filter_lists('spyware_block_list=', potential_conflicts_set)
        spyware_allow_lists= self.get_url_filter_lists('spyware_allow_list=', potential_conflicts_set)

        malware_profile_data = self.get_malware_profile_data(potential_conflicts_set)
        malware_alert_lists = self.get_malware_alert_list(potential_conflicts_set)
        malware_block_lists = self.get_malware_block_list(potential_conflicts_set)
        malware_allow_lists = self.get_malware_allow_list(potential_conflicts_set)

        url_block_lists = self.get_url_filter_lists('url_block_list=', potential_conflicts_set)
        url_alert_lists = self.get_url_filter_lists('url_alert_list=', potential_conflicts_set)
        url_allow_lists = self.get_url_filter_lists('url_allow_list=', potential_conflicts_set)
        url_override_lists = self.get_url_filter_lists('url_override_list=', potential_conflicts_set)
        url_none_lists = self.get_url_filter_lists('url_none_list=', potential_conflicts_set)

        log_events = self.get_url_filter_lists('log_events=', potential_conflicts_set)
        notifications = self.get_notifications(potential_conflicts_set)

        # TODO: add later when active times are mandatory
        # common_active_times = self.get_policy_active_times(intersecting_attr_set)
        # temp workaround
        common_active_times = [1]
        protocols_overlap=False
        ports_overlap=False

        if len(protocols)==1:
            protocols_overlap = True
            if 'any' in protocols: #1
                ports_overlap = True
            elif '6' in protocols: #TCP
                if tcp_dports is not None and tcp_sports is not None and len(tcp_dports)==1 and len(tcp_sports)==1:
                    ports_overlap = True
            elif '17' in protocols: #UDP
                if udp_dports is not None and udp_sports is not None and len(udp_dports)==1 and len(udp_sports)==1:
                    ports_overlap = True
            elif '1' in protocols: #ICMP
                    ports_overlap = True

        elif len(protocols)>1:
            if '6' in protocols and 'any' in protocols and '17' in protocols: # 6, 1, 17
                # one is an all-IP flow
                protocols_overlap=True
                ports_overlap=True

        if len(actions) > 1 and len(common_active_times)>0 and protocols_overlap and ports_overlap:
            conflict_sources.append('Access-control actions')
            return True

        if len(priorities) > 1 and len(common_active_times)>0 and protocols_overlap and ports_overlap:
            conflict_sources.append('QoS Priorities')
            return True

        if len(interfaces) > 1 and len(common_active_times)>0 and protocols_overlap and ports_overlap:
            return True

        if len(middlebox1_devices) > 1 and len(common_active_times)>0 and protocols_overlap and ports_overlap:
            return True

        if len(middlebox2_devices) > 1 and len(common_active_times)>0 and protocols_overlap and ports_overlap:
            return True

        # check if malware/spyware/url_filter profile data provided/populated
        if 'N/A' in malware_profiles and 'N/A' in spyware_profiles and \
            'N/A' in url_filter_profiles:
            # not provided
            if len(actions) > 1 and len(common_active_times)>0 and protocols_overlap and ports_overlap:
                conflict_sources.append('Access-control Actions')
                return True
        else:

            if len(common_active_times)>0 and len(actions)==1 and protocols_overlap and ports_overlap:
               if self.malware_profiles_conflict(malware_alert_lists, malware_block_lists, malware_allow_lists, malware_profiles, actions[0]):
                        conflict_sources.append('Malware-signature lists')
                        return True

               if self.url_filter_profiles_conflict(url_alert_lists, url_block_lists, url_allow_lists,
                                                     url_override_lists, url_none_lists, url_filter_profiles, actions[0]):
                    conflict_sources.append('URL-Filter lists')
                    return True

               if self.session_log_profiles_conflict(log_events):
                    conflict_sources.append('Session-log events')
                    return True

               if self.spyware_profiles_conflict(spyware_alert_lists, spyware_block_lists, spyware_allow_lists, spyware_profiles, actions[0]):
                    conflict_sources.append('Spyware-severity lists')
                    return True

               if len(notifications) > 1:
                   conflict_sources.append('IPS-notifications')
                   return True

        return False

    def edge_attributes_overlap(self, propositions):
        if propositions==None:
           raise MetagraphException('propositions',resources['value_null'])

        # currently checks if actions conflict
        # extend later to include active times etc
        actions= self.get_actions(propositions)
        protocols = self.get_protocols(propositions)
        tcp_dports = self.get_tcp_ports(propositions,True)
        udp_dports = self.get_udp_ports(propositions,True)
        tcp_sports = self.get_tcp_ports(propositions,False)
        udp_sports = self.get_udp_ports(propositions,False)

        malware_profiles= self.get_malware_profiles(propositions)
        url_filter_profiles= self.get_url_filter_profiles(propositions)
        spyware_profiles= self.get_spyware_profiles(propositions)

        spyware_alert_lists= self.get_url_filter_lists('spyware_alert_list=', propositions)
        spyware_block_lists= self.get_url_filter_lists('spyware_block_list=', propositions)
        spyware_allow_lists= self.get_url_filter_lists('spyware_allow_list=', propositions)

        malware_alert_lists = self.get_malware_alert_list(propositions)
        malware_block_lists = self.get_malware_block_list(propositions)
        malware_allow_lists = self.get_malware_allow_list(propositions)


        protocols_overlap=False
        ports_overlap=False

        if len(protocols)==1:
            protocols_overlap = True
            if 'any' in protocols: #1
                ports_overlap = True
            elif '6' in protocols: #TCP
                if tcp_dports is not None and tcp_sports is not None and len(tcp_dports)==1 and len(tcp_sports)==1:
                    ports_overlap = True
            elif '17' in protocols: #UDP
                if udp_dports is not None and udp_sports is not None and len(udp_dports)==1 and len(udp_sports)==1:
                    ports_overlap = True
            elif '1' in protocols: #ICMP
                    ports_overlap = True

        elif len(protocols)>1:
            if '6' in protocols and 'any' in protocols and '17' in protocols: # 6, 1, 17
                # one is an all-IP flow
                protocols_overlap=True
                ports_overlap=True

        if len(actions) == 1 and protocols_overlap and ports_overlap:
            return True

        # check if malware/spyware/url_filter profile data provided/populated
        if 'N/A' in malware_profiles and 'N/A' in spyware_profiles and \
            'N/A' in url_filter_profiles:
            # not provided
            if len(actions) == 1 and protocols_overlap and ports_overlap:
                return True

        return False


    def malware_profiles_conflict(self, malware_alert_sig_data, malware_block_sig_data, malware_allow_sig_data, malware_profiles, action):
        import ast

        malware_alert_sig_lists = []
        malware_block_sig_lists = []
        malware_allow_sig_lists = []

        if len(malware_profiles)==1: return False

        try:

            for alert_sig_data in malware_alert_sig_data:
                if alert_sig_data!='{}' and alert_sig_data!='[]':
                    data = ast.literal_eval(alert_sig_data)
                    if data not in malware_alert_sig_lists:
                        malware_alert_sig_lists.append(data)
                else:
                    if dict() not in malware_alert_sig_lists:
                        malware_alert_sig_lists.append(dict())

            for block_sig_data in malware_block_sig_data:
                if block_sig_data!='{}' and block_sig_data!='[]':
                    data = ast.literal_eval(block_sig_data)
                    if data not in malware_block_sig_lists:
                        malware_block_sig_lists.append(data)
                else:
                    if dict() not in malware_block_sig_lists:
                        malware_block_sig_lists.append(dict())

            for allow_sig_data in malware_allow_sig_data:
                if allow_sig_data!='{}' and allow_sig_data!='[]':
                    data = ast.literal_eval(allow_sig_data)
                    if data not in malware_allow_sig_lists:
                        malware_allow_sig_lists.append(data)
                else:
                    if dict() not in malware_allow_sig_lists:
                        malware_allow_sig_lists.append(dict())

            # check if different types of signature lists overlap
            if self.malware_signature_lists_overlap(malware_alert_sig_lists, malware_block_sig_lists):
                return True

            if self.malware_signature_lists_overlap(malware_alert_sig_lists, malware_allow_sig_lists):
                return True

            if self.malware_signature_lists_overlap(malware_block_sig_lists, malware_allow_sig_lists):
                return True

            # check if there are distinct sig lists of same type
            if self.distinct_malware_signature_lists(malware_alert_sig_lists, action):
                return True

            if self.distinct_malware_signature_lists(malware_block_sig_lists, action):
                return True

            if self.distinct_malware_signature_lists(malware_allow_sig_lists, action, True):
                return True


        except BaseException, e:
            pass
            #print('error::malware_profiles_conflict: %s'%str(e))

        return False

    def spyware_profiles_conflict(self, spyware_alert_data, spyware_block_data, spyware_allow_data, spyware_profiles, action):
        import ast

        spyware_alert_lists = []
        spyware_block_lists = []
        spyware_allow_lists = []

        if len(spyware_profiles)==1: return False

        for alert_data in spyware_alert_data:
            if alert_data!='{}' and alert_data!='[]':
                data = ast.literal_eval(alert_data)
                if data not in spyware_alert_lists:
                    spyware_alert_lists.append(data)
            else:
                if dict() not in spyware_alert_lists:
                    spyware_alert_lists.append(dict())

        for block_data in spyware_block_data:
            if block_data!='{}' and block_data!='[]':
                data = ast.literal_eval(block_data)
                if data not in spyware_block_lists:
                    spyware_block_lists.append(data)
            else:
                if dict() not in spyware_block_lists:
                    spyware_block_lists.append(dict())

        for allow_data in spyware_allow_data:
            if allow_data!='{}' and allow_data!='[]':
                data = ast.literal_eval(allow_data)
                if data not in spyware_allow_lists:
                    spyware_allow_lists.append(data)
            else:
                if dict() not in spyware_allow_lists:
                    spyware_allow_lists.append(dict())

        # check if different types of severity lists overlap
        if self.spyware_severity_lists_overlap(spyware_alert_lists, spyware_block_lists):
            return True

        if self.spyware_severity_lists_overlap(spyware_alert_lists, spyware_allow_lists):
            return True

        if self.spyware_severity_lists_overlap(spyware_block_lists, spyware_allow_lists):
            return True

        # check if there are distinct severity lists of same type
        if self.distinct_spyware_severity_lists(spyware_alert_lists, action):
            return True

        if self.distinct_spyware_severity_lists(spyware_block_lists, action):
            return True

        if self.distinct_spyware_severity_lists(spyware_allow_lists, action, True):
            return True

        return False

    def url_filter_profiles_conflict(self, url_alert_data, url_block_data, url_allow_data, url_override_data, url_none_data, url_filter_profiles, action):

        if len(url_filter_profiles)==1: return False

        url_alert_lists= self.get_url_lists(url_alert_data)
        url_block_lists= self.get_url_lists(url_block_data)
        url_allow_lists= self.get_url_lists(url_allow_data)
        url_override_lists= self.get_url_lists(url_override_data)
        url_none_lists= self.get_url_lists(url_none_data)

        # check if different types of url filter lists overlap
        # overlap with none_list does not cause any conflicts
        if self.url_filter_lists_overlap(url_alert_lists, url_block_lists):
            return True

        if self.url_filter_lists_overlap(url_alert_lists, url_allow_lists):
            return True

        if self.url_filter_lists_overlap(url_alert_lists, url_override_lists):
            return True

        if self.url_filter_lists_overlap(url_block_lists, url_allow_lists):
            return True

        if self.url_filter_lists_overlap(url_block_lists, url_override_lists):
            return True

        if self.url_filter_lists_overlap(url_allow_lists, url_override_lists):
            return True

        # check if distinct url filter lists of same type exist
        if self.distinct_url_filter_lists(url_alert_lists, action):
            return True

        if self.distinct_url_filter_lists(url_block_lists, action):
            return True

        if self.distinct_url_filter_lists(url_allow_lists, action, True):
            return True

        if self.distinct_url_filter_lists(url_override_lists, action, True):
            return True

        return False

    def session_log_profiles_conflict(self, log_events):

        return self.distinct_log_event_lists(log_events)

    def url_filter_lists_overlap(self, filter_list1, filter_list2):

        for list1 in filter_list1:
            for list2 in filter_list2:
                intersection = set(list1).intersection(set(list2))
                if intersection is not None:
                    return True

        return False

    def malware_signature_lists_overlap(self, sig_list1, sig_list2):
        for elt1 in sig_list1:
            for elt2 in sig_list2:
                apps_elt1 = set(elt1.keys())
                apps_elt2 = set(elt2.keys())
                apps_intersection = apps_elt1.intersection(apps_elt2)
                if apps_intersection is not None:
                    for app in list(apps_intersection):
                         sigs1 = set(sig_list1[app])
                         sigs2 = set(sig_list2[app])
                         intersection = sigs1.intersection(sigs2)
                         if intersection is not None:
                             return True


        return False

    def spyware_severity_lists_overlap(self, severity_list1, severity_list2):
        for elt1 in severity_list1:
            for elt2 in severity_list2:
                severity1 = set(elt1.keys())
                severity2 = set(elt2.keys())
                intersection = severity1.intersection(severity2)
                if intersection is not None:
                    for severity in list(intersection):
                         threat_names1 = severity_list1[severity]['threat-name']
                         threat_names2 = severity_list2[severity]['threat-name']
                         intersection1 = set(threat_names1).intersection(set(threat_names2))

                         threat_categories1 = severity_list2[severity]['threat-category']
                         threat_categories2 = severity_list2[severity]['threat-category']

                         intersection2 = set(threat_categories1).intersection(set(threat_categories2))

                         if intersection1 is not None and intersection2 is not None:
                             return True


        return False

    def distinct_malware_signature_lists(self, sig_lists, action, check_action_allow=False):

        for sig_list1 in sig_lists:
            for sig_list2 in sig_lists:
                if sig_list1 != sig_list2:
                    # case when one list is empty and one is not
                    if (sig_list1 is None or len(sig_list1)==0) and \
                       (sig_list2 is not None and len(sig_list2) > 0):
                       if check_action_allow:
                           # additional criteria for allow list
                           if action!='1':
                              return True
                       else:
                           return True

                    elif (sig_list2 is None or len(sig_list2)==0) and \
                       (sig_list1 is not None and len(sig_list1) > 0):
                       if check_action_allow:
                           # additional criteria for allow list
                           if action!='1':
                              return True
                       else:
                           return True

                    # case when both lists are non empty
                    apps1 = set(sig_list1.keys())
                    apps2 = set(sig_list2.keys())
                    apps_intersection = apps1.intersection(apps2)
                    for app in apps_intersection:
                        sigs1 = set(sig_list1[app])
                        sigs2 = set(sig_list2[app])
                        if sigs1!=sigs2:
                            return True

        return False

    def distinct_spyware_severity_lists(self, severity_lists, action, check_action_allow=False):

        for list1 in severity_lists:
            for list2 in severity_lists:
                if list1 != list2:
                    # case when one list is empty and one is not
                    if (list1 is None or len(list1)==0) and \
                       (list2 is not None and len(list2) > 0):
                       if check_action_allow:
                           # additional criteria for allow list
                           if action!='1':
                              return True
                       else:
                           return True

                    elif (list2 is None or len(list2)==0) and \
                       (list1 is not None and len(list1) > 0):
                       if check_action_allow:
                           # additional criteria for allow list
                           if action!='1':
                              return True
                       else:
                           return True

                    # case when both lists are non empty
                    severities1 = set(list1.keys())
                    severities2 = set(list2.keys())
                    sev_intersection = severities1.intersection(severities2)

                    for severity in sev_intersection:
                        threat_names1 = list1[severity]['threat-name']
                        threat_names2 = list2[severity]['threat-name']
                        threat_categories1 = list2[severity]['threat-category']
                        threat_categories2 = list2[severity]['threat-category']

                        if set(threat_names1)!=set(threat_names2):
                            return True
                        if set(threat_categories1)!=set(threat_categories2):
                            return True

        return False

    def distinct_url_filter_lists(self, filter_lists, action, check_action_allow=False):

        for list1 in filter_lists:
            for list2 in filter_lists:
                if list1!=list2:
                    # check if only one list is empty
                    if (list1 is None or len(list1)==0) and \
                       (list2 is not None and len(list2) > 0):
                         if check_action_allow:
                           # additional criteria for allow list
                           if action!='1':
                              return True
                         else:
                           return True

                    elif (list2 is None or len(list2)==0) and \
                         (list1 is not None and len(list1) > 0):
                         if check_action_allow:
                           # additional criteria for allow list
                           if action!='1':
                              return True
                         else:
                           return True

                    # both lists are non empty
                    if (list1 is not None and len(list1) >0) and \
                       (list2 is not None and len(list2) >0) and \
                       set(list1)!=set(list2):
                        return True


        return False

    def distinct_log_event_lists(self, log_event_lists):
        for list1 in log_event_lists:
            for list2 in log_event_lists:
                if list1!=list2:
                    # check if only one list is empty
                    if (list1 is None or len(list1)==0) and \
                       (list2 is not None and len(list2) >0) :
                        return True
                    elif (list2 is None or len(list2)==0) and \
                       (list1 is not None and len(list1) >0) :
                        return True

                    # both lists are non empty
                    if (list1 is not None and len(list1) >0) and \
                       (list2 is not None and len(list2) >0) and \
                       set(list1)!=set(list2):
                        return True

        return False

    def get_url_lists(self, url_data):
        url_lists=[]

        for data in url_data:
            if data!='{}' and data!='[]':
                data = data.replace('[','')
                data = data.replace(']','')
                items = data.split(',')
                if items not in url_lists:
                    url_lists.append(items)
            else:
                if [] not in url_lists:
                    url_lists.append([])

        return url_lists

    def get_protocols(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])

        protocols=[]
        for attribute in attributes:
            if 'protocol=' in attribute:
                value= attribute.replace('protocol=','')
                if value not in protocols:
                    protocols.append(value)
        return protocols

    def get_tcp_ports(self, attributes, dport):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])

        tcp_ports=[]
        for attribute in attributes:
            if dport and 'TCP.dport=' in attribute:
                value= attribute.replace('TCP.dport=','')
                #if value not in tcp_ports:
                # allow duplicates
                tcp_ports.append(value)
            elif not dport and 'TCP.sport=' in attribute:
                value= attribute.replace('TCP.sport=','')
                #if value not in tcp_ports:
                # allow duplicates
                tcp_ports.append(value)
        result = list(set(tcp_ports))
        if '0-65535' in result and len(result)>1:
            result.remove('0-65535')

        return result

    def get_udp_ports(self, attributes, dport):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])

        udp_ports=[]
        for attribute in attributes:
            if dport and 'UDP.dport=' in attribute:
                value= attribute.replace('UDP.dport=','')
                #if value not in udp_ports:
                # allow duplicates
                udp_ports.append(value)
            elif not dport and 'UDP.sport=' in attribute:
                value= attribute.replace('UDP.sport=','')
                #if value not in udp_ports:
                # allow duplicates
                udp_ports.append(value)

        result = list(set(udp_ports))
        if '0-65535' in result and len(result)>1:
            result.remove('0-65535')

        return result

    def get_malware_profiles(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        profiles=[]
        for attribute in attributes:
            if 'profile_malware=' in attribute:
                value= attribute.replace('profile_malware=','')
                if value not in profiles:
                    profiles.append(value)
        return profiles

    def get_malware_signature_lists(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        sig_lists=[]
        for attribute in attributes:
            if 'malware_signature_list=' in attribute:
                value= attribute.replace('malware_signature_list=','')
                if value not in sig_lists:
                    sig_lists.append(value)
        return sig_lists

    def get_malware_app_lists(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        app_lists=[]
        for attribute in attributes:
            if 'malware_app_list=' in attribute:
                value= attribute.replace('malware_app_list=','')
                if value not in app_lists:
                    app_lists.append(value)
        return app_lists

    def get_malware_profile_data(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        profile_data=[]
        for attribute in attributes:
            if 'malware_profile_details=' in attribute:
                value= attribute.replace('malware_profile_details=','')
                if value not in profile_data:
                    profile_data.append(value)
        return profile_data

    def get_malware_alert_list(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        alert_list=[]
        for attribute in attributes:
            if 'malware_alert_list=' in attribute:
                value= attribute.replace('malware_alert_list=','')
                if value not in alert_list:
                    alert_list.append(value)
        return alert_list

    def get_malware_allow_list(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        allow_list=[]
        for attribute in attributes:
            if 'malware_allow_list=' in attribute:
                value= attribute.replace('malware_allow_list=','')
                if value not in allow_list:
                    allow_list.append(value)
        return allow_list

    def get_malware_block_list(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        block_list=[]
        for attribute in attributes:
            if 'malware_block_list=' in attribute:
                value= attribute.replace('malware_block_list=','')
                if value not in block_list:
                    block_list.append(value)
        return block_list

    def get_spyware_severity_lists(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        sev_lists=[]
        for attribute in attributes:
            if 'spyware_severity_list=' in attribute:
                value= attribute.replace('spyware_severity_list=','')
                if value not in sev_lists:
                    sev_lists.append(value)
        return sev_lists

    def get_spyware_profiles(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        profiles=[]
        for attribute in attributes:
            if 'profile_spyware=' in attribute:
                value= attribute.replace('profile_spyware=','')
                if value not in profiles:
                    profiles.append(value)
        return profiles

    def get_url_filter_profiles(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        profiles=[]
        for attribute in attributes:
            if 'profile_url_filter=' in attribute:
                value= attribute.replace('profile_url_filter=','')
                if value not in profiles:
                    profiles.append(value)
        return profiles

    def get_url_block_lists(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        block_lists=[]
        for attribute in attributes:
            if 'url_block_list=' in attribute:
                value= attribute.replace('url_block_list=','')
                if value not in block_lists:
                    block_lists.append(value)
        return block_lists

    def get_url_alert_lists(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        alert_lists=[]
        for attribute in attributes:
            if 'url_alert_list=' in attribute:
                value= attribute.replace('url_alert_list=','')
                if value not in alert_lists:
                    alert_lists.append(value)
        return alert_lists

    def get_url_filter_lists(self, attribute_key, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        result=[]
        for attribute in attributes:
            if attribute_key in attribute:
                value= attribute.replace(attribute_key,'')
                if value not in result:
                    result.append(value)
        return result

    def get_log_events(self, attributes):
        if attributes==None:
            raise MetagraphException('attributes', resources['value_null'])
        profiles=[]
        for attribute in attributes:
            if 'log_event=' in attribute:
                value=attribute.replace('log_event=','')
                if value!='None' and value not in profiles:
                    profiles.append(value)
        return profiles

    def get_notifications(self, attributes):
        if attributes==None:
            raise MetagraphException('attributes', resources['value_null'])
        profiles=[]
        for attribute in attributes:
            if 'notification=' in attribute:
                value=attribute.replace('notification=','')
                if value not in profiles:
                    profiles.append(value)
        return profiles

    def get_priorities(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        priorities=[]
        for attribute in attributes:
            if 'priority=' in attribute:
                value= attribute.replace('priority=','')
                if value not in priorities:
                    priorities.append(value)
        return priorities

    def get_interfaces(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        interfaces=[]
        for attribute in attributes:
            if 'interface=' in attribute:
                value= attribute.replace('interface=','')
                if value not in interfaces:
                    interfaces.append(value)
        return interfaces

    def get_middleboxes(self, attributes, middlebox_id):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        middleboxes = []
        middlebox_attr = 'middlebox%s=' % middlebox_id
        for attribute in attributes:
            if middlebox_attr in attribute:
                value= attribute.replace(middlebox_attr,'')
                if value not in middleboxes:
                    middleboxes.append(value)
        return middleboxes

    def get_policy_active_times(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        active_times=[]
        for attribute in attributes:
            if 'active_time=' in attribute:
                value= attribute.replace('active_time=','')
                if value not in active_times:
                    active_times.append(value)
        return active_times

    def get_active_times(self, attributes):
        if attributes==None:
           raise MetagraphException('attributes',resources['value_null'])
        active_times=[]
        max_mins=60*24
        for attribute in attributes:
            if 'time' in attribute:
                temp= attribute.replace('time','')
                gt=False
                lt=False
                eq=False
                if '>'in temp:
                    gt=True
                    temp=temp.replace('>','')
                if '=' in temp:
                    eq=True
                    temp=temp.replace('=','')
                if '<' in temp:
                    lt=True
                    temp=temp.replace('<','')

                hours_comp= math.floor(float(temp))
                mins_comp=(float(temp)-hours_comp)*60
                total_mins = 60*hours_comp + mins_comp
                if eq and (not lt and not gt) and [total_mins,total_mins] not in active_times:
                    active_times.append([total_mins,total_mins])
                elif eq and lt and [0,total_mins] not in active_times:
                    active_times.append([0,total_mins])
                elif eq and gt and [total_mins,max_mins] not in active_times:
                    active_times.append([total_mins,max_mins])
                elif lt and [0,total_mins-1] not in active_times:
                    active_times.append([0,total_mins-1])
                elif gt and [total_mins+1,max_mins] not in active_times:
                    active_times.append([total_mins+1,max_mins])

        return active_times

    @staticmethod
    def get_actions(attributes):
        """ Filters the given list of attributes and returns a list of action-attribute values.
        :param attributes:  list
        :return: list of strings
        """

        if attributes is None:
            raise MetagraphException('attributes', resources['value_null'])
        actions = []
        for attribute in attributes:
            if 'action' in attribute:
                value = attribute.replace('action=', '')
                if value not in actions:
                    actions.append(value)
        return actions

    @staticmethod
    def get_malware_sigs(attributes):
        if attributes is None:
            raise MetagraphException('attributes', resources['value_null'])
        sigs = []
        for attribute in attributes:
            if 'malware_sigs' in attribute:
                value = attribute.replace('malware_sigs=', '')
                value = value.replace('[','')
                value = value.replace(']','')
                value = value.split(',')
                if value not in sigs:
                    sigs.append(value)
        return sigs

    @staticmethod
    def get_sig_present(attributes):
        if attributes is None:
            raise MetagraphException('attributes', resources['value_null'])
        present = []
        for attribute in attributes:
            if 'sig_present' in attribute:
                value = attribute.replace('sig_present=', '')
                if value not in present:
                    present.append(value)
        return present

    def is_connected(self, source, target, logical_expressions, interpretations):
        """Checks if subset1 is connected to subset2.
        :param source: set
        :param target: set
        :param logical_expressions: list of strings
        :param interpretations: lists of tuples
        :return: boolean
        """

        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])
        if logical_expressions is None or len(logical_expressions) == 0:
            raise MetagraphException('logical_expressions', resources['value_null'])
        if interpretations is None or len(interpretations) == 0:
            raise MetagraphException('interpretations', resources['value_null'])
        if not source.issubset(self.variables_set):
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.issubset(self.variables_set):
            raise MetagraphException('target', resources['not_a_subset'])

        # check expressions are over X_p
        for logical_expression in logical_expressions:
            logical_expression_copy = copy.copy(logical_expression)
            logical_expression_copy = logical_expression_copy.replace('.', ' ')
            logical_expression_copy = logical_expression_copy.replace('|', ' ')
            logical_expression_copy = logical_expression_copy.replace('!', ' ')
            logical_expression_copy = logical_expression_copy.replace('(', '')
            logical_expression_copy.replace(')', '')
            items = logical_expression_copy.split(' ')
            for item in items:
                item = item.replace(' ', '')
                if item != '' and item not in self.propositions_set:
                    raise MetagraphException('logical_expression', resources['arguments_invalid'])

        # check metapath exists for at least one interpretation
        for interpretation in interpretations:
            true_propositions = []
            false_propositions = []
            for tuple_elt in interpretation:
                if tuple_elt[0] not in self.propositions_set:
                    raise MetagraphException('interpretations', resources['arguments_invalid'])
                if tuple_elt[1] and tuple_elt[0] not in true_propositions:
                    true_propositions.append(tuple_elt[0])
                elif tuple_elt[0] not in true_propositions:
                    false_propositions.append(tuple_elt[0])

            # compute context metagraph
            context = self.get_context(true_propositions, false_propositions)
            metapaths = context.get_all_metapaths_from(source, target)

            if metapaths is not None and len(metapaths) >= 1:
                return True

        return False

    def is_fully_connected(self, source, target, logical_expressions, interpretations):
        """Checks if subset1 is fully connected to subset2.
        :param source: set
        :param target: set
        :param logical_expressions: list of strings
        :param interpretations: lists of tuples
        :return: boolean
        """

        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])
        if logical_expressions is None or len(logical_expressions) == 0:
            raise MetagraphException('logical_expressions', resources['value_null'])
        if interpretations is None or len(interpretations) == 0:
            raise MetagraphException('interpretations', resources['value_null'])
        if not source.issubset(self.variables_set):
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.issubset(self.variables_set):
            raise MetagraphException('target', resources['not_a_subset'])

        # check expressions are over X_p
        for logical_expression in logical_expressions:
            logical_expression_copy = copy.copy(logical_expression)
            logical_expression_copy = logical_expression_copy.replace('.', ' ')
            logical_expression_copy = logical_expression_copy.replace('|', ' ')
            logical_expression_copy = logical_expression_copy.replace('!', ' ')
            logical_expression_copy = logical_expression_copy.replace('(', '')
            logical_expression_copy.replace(')', '')
            items = logical_expression_copy.split(' ')
            for item in items:
                item = item.replace(' ', '')
                if item != '' and item not in self.propositions_set:
                    raise MetagraphException('logical_expression', resources['arguments_invalid'])

        # check metapath exists for every interpretation
        for interpretation in interpretations:
            true_propositions = []
            false_propositions = []
            for tuple_elt in interpretation:
                if tuple_elt[0] not in self.propositions_set:
                    raise MetagraphException('interpretations', resources['arguments_invalid'])
                if tuple_elt[1] and tuple_elt[0] not in true_propositions:
                    true_propositions.append(tuple_elt[0])
                elif tuple_elt[0] not in true_propositions:
                    false_propositions.append(tuple_elt[0])

            # compute context metagraph
            context = self.get_context(true_propositions, false_propositions)
            metapaths = context.get_all_metapaths_from(source, target)

            if not(metapaths is not None and len(metapaths) >= 1):
                return False

        return True

    def is_redundantly_connected(self, source, target, logical_expressions, interpretations):
        """Checks if subset1 is non-redundantly connected to subset2.
        :param source: set
        :param target: set
        :param logical_expressions: list of strings
        :param interpretations: lists of tuples
        :return: boolean
        """

        if source is None or len(source) == 0:
            raise MetagraphException('source', resources['value_null'])
        if target is None or len(target) == 0:
            raise MetagraphException('target', resources['value_null'])
        if logical_expressions is None or len(logical_expressions) == 0:
            raise MetagraphException('logical_expressions', resources['value_null'])
        if interpretations is None or len(interpretations) == 0:
            raise MetagraphException('interpretations', resources['value_null'])
        if not source.issubset(self.variables_set):
            raise MetagraphException('source', resources['not_a_subset'])
        if not target.issubset(self.variables_set):
            raise MetagraphException('target', resources['not_a_subset'])

        # check expressions are over X_p
        for logical_expression in logical_expressions:
            logical_expression_copy = copy.copy(logical_expression)
            logical_expression_copy = logical_expression_copy.replace('.', ' ')
            logical_expression_copy = logical_expression_copy.replace('|', ' ')
            logical_expression_copy = logical_expression_copy.replace('!', ' ')
            logical_expression_copy = logical_expression_copy.replace('(', '')
            logical_expression_copy.replace(')', '')
            items = logical_expression_copy.split(' ')
            for item in items:
                item = item.replace(' ', '')
                if item != '' and item not in self.propositions_set:
                    raise MetagraphException('logical_expression', resources['arguments_invalid'])

        # check metapath exists for every interpretation
        for interpretation in interpretations:
            true_propositions = []
            false_propositions = []
            for tuple_elt in interpretation:
                if tuple_elt[0] not in self.propositions_set:
                    raise MetagraphException('interpretations', resources['arguments_invalid'])
                if tuple_elt[1] and tuple_elt[0] not in true_propositions:
                    true_propositions.append(tuple_elt[0])
                elif tuple_elt[0] not in true_propositions:
                    false_propositions.append(tuple_elt[0])

            # compute context metagraph
            context = self.get_context(true_propositions, false_propositions)
            metapaths = context.get_all_metapaths_from(source, target)

            if metapaths is not None and len(metapaths) > 1:
                return False

        return True

    def is_non_redundant(self, logical_expressions, interpretations):
        """ Checks if a conditional metagraph is non redundant.
        :param logical_expressions: list of strings
        :param interpretations: lists of tuples
        :return: boolean
        """

        if logical_expressions is None or len(logical_expressions) == 0:
            raise MetagraphException('logical_expressions', resources['value_null'])
        if interpretations is None or len(interpretations) == 0:
            raise MetagraphException('interpretations', resources['value_null'])

        # check expressions are over X_p
        for logical_expression in logical_expressions:
            logical_expression_copy = copy.copy(logical_expression)
            logical_expression_copy = logical_expression_copy.replace('.', ' ')
            logical_expression_copy = logical_expression_copy.replace('|', ' ')
            logical_expression_copy = logical_expression_copy.replace('!', ' ')
            logical_expression_copy = logical_expression_copy.replace('(', '')
            logical_expression_copy.replace(')', '')
            items = logical_expression_copy.split(' ')
            for item in items:
                item = item.replace(' ', '')
                if item != '' and item not in self.propositions_set:
                    raise MetagraphException('logical_expression', resources['arguments_invalid'])

        # check metapath exists for at least one interpretation
        for interpretation in interpretations:
            true_propositions = []
            false_propositions = []
            for tuple_elt in interpretation:
                if tuple_elt[0] not in self.propositions_set:
                    raise MetagraphException('interpretations', resources['arguments_invalid'])
                if tuple_elt[1] and tuple_elt[0] not in true_propositions:
                    true_propositions.append(tuple_elt[0])
                elif tuple_elt[0] not in true_propositions:
                    false_propositions.append(tuple_elt[0])

            # compute context metagraph
            context = self.get_context(true_propositions, false_propositions)

            for x in self.variables_set:
                edge_list = []
                for edge in context.edges:
                    if x in edge.outvertex and edge not in edge_list:
                        edge_list.append(edge)
                if len(edge_list) > 1:
                    return False

            return True

    def get_associated_edges(self, element):
         return [edge for edge in self.edges if element in edge.invertex]

    def get_protocol_propositions(self, original_propositions):
        result=[]
        for prop in list(original_propositions):
            if 'protocol' in prop:
               result.append(prop)

        return set(result)

    def get_port_propositions(self, original_propositions):
        result=[]
        for prop in list(original_propositions):
            if 'dport' in prop:
                result.append(prop)

        return set(result)

    def get_sport_propositions(self, original_propositions):
        result=[]
        for prop in list(original_propositions):
            if 'sport' in prop:
                result.append(prop)

        return set(result)

    def check_conflicts(self):

        result=[]
        potentially_conflicting_edges1=dict()
        potentially_conflicting_edges2=dict()

        for edge1 in self.edges:
            for edge2 in self.edges:
                if not MetagraphHelper().are_edges_equal(edge1,edge2):
                   invertex1 = edge1.invertex.difference(self.propositions_set)
                   outvertex1 = edge1.outvertex.difference(self.propositions_set)
                   invertex2 = edge2.invertex.difference(self.propositions_set)
                   outvertex2 = edge2.outvertex.difference(self.propositions_set)

                   invertex_int= invertex1.intersection(invertex2)
                   outvertex_int= outvertex1.intersection(outvertex2)

                   if (invertex_int is not None and len(invertex_int)>0):
                      # invertex EPGs overlap
                      # check if flows overlap
                      flow_overlap=False
                      prot_propositions1 = self.get_protocol_propositions(edge1.invertex)
                      prot_propositions2 = self.get_protocol_propositions(edge2.invertex)
                      prot_int = prot_propositions1.intersection(prot_propositions2)

                      if (prot_int is not None and len(prot_int)>0):
                         # protocols overlap
                         if 'protocol=any' not in prot_int: #1
                             # TCP or UDP flow -check port overlap
                             port_propositions1 = self.get_port_propositions(edge1.invertex)
                             port_propositions2 = self.get_port_propositions(edge2.invertex)
                             port_int = port_propositions1.intersection(port_propositions2)
                             if (port_int is not None and len(port_int)>0):
                                # port overlap
                                flow_overlap=True
                         else:
                             flow_overlap=True

                      if flow_overlap:
                          # check if already accounted
                          if edge2 in potentially_conflicting_edges1 and \
                             edge1 in potentially_conflicting_edges1[edge2]:
                              continue

                          if edge1 not in potentially_conflicting_edges1:
                            potentially_conflicting_edges1[edge1]=[]

                          if edge2 not in potentially_conflicting_edges1[edge1]:
                            potentially_conflicting_edges1[edge1].append(edge2)


                   if (outvertex_int is not None and len(outvertex_int)>0):
                      # outvertex EPGs overlap
                      # check if flows overlap
                      flow_overlap=False
                      prot_propositions1 = self.get_protocol_propositions(edge1.invertex)
                      prot_propositions2 = self.get_protocol_propositions(edge2.invertex)
                      prot_int = prot_propositions1.intersection(prot_propositions2)

                      if (prot_int is not None and len(prot_int)>0):
                         # protocols overlap
                         if 'protocol=any' not in prot_int: #1
                             # TCP or UDP flow -check port overlap
                             port_propositions1 = self.get_port_propositions(edge1.invertex)
                             port_propositions2 = self.get_port_propositions(edge2.invertex)
                             port_int = port_propositions1.intersection(port_propositions2)
                             if (port_int is not None and len(port_int)>0):
                                # port overlap
                                flow_overlap=True
                         else:
                             flow_overlap=True

                      if flow_overlap:
                          # check if already accounted
                          if edge2 in potentially_conflicting_edges2 and \
                             edge1 in potentially_conflicting_edges2[edge2]:
                              continue

                          if edge1 not in potentially_conflicting_edges2:
                            potentially_conflicting_edges2[edge1]=[]

                          if edge2 not in potentially_conflicting_edges2[edge1]:
                            potentially_conflicting_edges2[edge1].append(edge2)

        #print('step1')
        # determine metapath sources and targets
        processed=[]
        for edge1, value in potentially_conflicting_edges1.iteritems():
            source_set = edge1.invertex.difference(self.propositions_set)
            for edge2 in potentially_conflicting_edges1[edge1]:
               source_set = source_set.union(edge2.invertex.difference(self.propositions_set))

            for edge3, value in potentially_conflicting_edges2.iteritems():
                if len(potentially_conflicting_edges2[edge3])< 1:
                    continue
                target_set = edge3.outvertex

                # check if edges apply to overlapping flows
                flow_overlap=False
                prot_propositions1 = self.get_protocol_propositions(edge1.invertex)
                prot_propositions3 = self.get_protocol_propositions(edge3.invertex)
                prot_int = prot_propositions1.intersection(prot_propositions3)

                if (prot_int is not None and len(prot_int)>0):
                  # protocols overlap
                  if 'protocol=any' not in prot_int: #1
                     # TCP or UDP flow -check port overlap
                     port_propositions1 = self.get_port_propositions(edge1.invertex)
                     port_propositions3 = self.get_port_propositions(edge3.invertex)
                     port_int = port_propositions1.intersection(port_propositions3)
                     if (port_int is not None and len(port_int)>0):
                        # port overlap
                        flow_overlap=True
                  else:
                     flow_overlap=True

                if flow_overlap:
                   if (str(list(source_set)), str(list(target_set)))  in processed:
                       continue
                   processed.append((str(list(source_set)), str(list(target_set))))
                   mps= self.get_all_metapaths_from(source_set, target_set, True)
                   if mps is not None and len(mps)>0:
                       for mp in mps:
                          if mp not in result:
                             result.append(mp)

        #print('step2')
        #print('returning result..')
        if len(result)==0:
            #print('NO conflicts found')
            return None

        #print('inconsistencies found')
        potential_conflicts = result
        conflicts_final=None
        if potential_conflicts is not None and len(potential_conflicts)>0:
            conflicting_metapaths=[]
            self.conflict_source_lookup=dict()
            for mp in potential_conflicts:
                conflicting_edges=[]
                #conflict_sources=[]
                if self.has_conflicts(mp, conflicting_edges=conflicting_edges):
                   self.conflict_source_lookup[mp]=conflicting_edges
                   conflicting_metapaths.append(mp)
                #else:
                #    print "Policy inconsistency Detected for metapath::"
                #    for edge in mp.edge_list:
                #       print "edge- %s"%(edge)
                #       #self.print_original_policy(edge)
                #    #pass

            filtered= MetagraphHelper().remove_duplicates(conflicting_metapaths)
            if len(filtered)==0:
                #print('NO conflicts found')
                return None
            #else:
            #    print('conflicts found found')

            if False:
                count=1
                conflicts_final = filtered
                for mp in filtered:
                       conflicting_edges = self.conflict_source_lookup[mp]
                       for edge_tuple in conflicting_edges: #mp.edge_list:
                           conflict_source_desc= '.'.join(edge_tuple[2])
                           #print "%s. Policy Conflicts Detected:: cause- %s"%(count,conflict_source_desc)
                           count+=1
                           self.print_original_policy(edge_tuple[0])
                           self.print_original_policy(edge_tuple[1])

            #print(' ')
            #print('-------------------------------------------------')


        return conflicts_final

    def check_redundancies(self):

        result=[]
        potentially_redundant_edges1=dict()
        potentially_redundant_edges2=dict()

        for edge1 in self.edges:
            for edge2 in self.edges:
                if edge1!=edge2:
                   invertex1 = edge1.invertex.difference(self.propositions_set)
                   outvertex1 = edge1.outvertex.difference(self.propositions_set)
                   invertex2 = edge2.invertex.difference(self.propositions_set)
                   outvertex2 = edge2.outvertex.difference(self.propositions_set)

                   invertex_int= invertex1.intersection(invertex2)
                   outvertex_int= outvertex1.intersection(outvertex2)

                   if (invertex_int is not None and len(invertex_int)>0):
                      # invertex EPGs overlap
                      # check if flows overlap
                      flow_overlap=False
                      prot_propositions1 = self.get_protocol_propositions(edge1.invertex)
                      prot_propositions2 = self.get_protocol_propositions(edge2.invertex)
                      prot_int = prot_propositions1.intersection(prot_propositions2)

                      if (prot_int is not None and len(prot_int)>0):
                         # protocols overlap
                         if 'protocol=1' not in prot_int:
                             # TCP or UDP flow -check port overlap
                             dport_propositions1 = self.get_port_propositions(edge1.invertex)
                             dport_propositions2 = self.get_port_propositions(edge2.invertex)
                             sport_propositions1 = self.get_sport_propositions(edge1.invertex)
                             sport_propositions2 = self.get_sport_propositions(edge2.invertex)
                             dport_int = dport_propositions1.intersection(dport_propositions2)
                             sport_int = sport_propositions1.intersection(sport_propositions2)
                             if (dport_int is not None and len(dport_int)>0) and (sport_int is not None and len(sport_int)>0):
                                # port overlap
                                flow_overlap=True
                         else:
                             flow_overlap=True

                      if flow_overlap:
                          # check if already accounted
                          if edge2 in potentially_redundant_edges1 and \
                             edge1 in potentially_redundant_edges1[edge2]:
                              continue

                          if edge1 not in potentially_redundant_edges1:
                            potentially_redundant_edges1[edge1]=[]

                          if edge2 not in potentially_redundant_edges1[edge1]:
                            potentially_redundant_edges1[edge1].append(edge2)

                   if (outvertex_int is not None and len(outvertex_int)>0):
                      # outvertex EPGs overlap
                      # check if flows overlap
                      flow_overlap=False
                      prot_propositions1 = self.get_protocol_propositions(edge1.invertex)
                      prot_propositions2 = self.get_protocol_propositions(edge2.invertex)
                      prot_int = prot_propositions1.intersection(prot_propositions2)

                      if (prot_int is not None and len(prot_int)>0):
                         # protocols overlap
                         if 'protocol=1' not in prot_int:
                             # TCP or UDP flow -check port overlap
                             dport_propositions1 = self.get_port_propositions(edge1.invertex)
                             dport_propositions2 = self.get_port_propositions(edge2.invertex)
                             sport_propositions1 = self.get_sport_propositions(edge1.invertex)
                             sport_propositions2 = self.get_sport_propositions(edge2.invertex)
                             dport_int = dport_propositions1.intersection(dport_propositions2)
                             sport_int = sport_propositions1.intersection(sport_propositions2)
                             if (dport_int is not None and len(dport_int)>0) and (sport_int is not None and len(sport_int)>0):
                                # port overlap
                                flow_overlap=True
                         else:
                             flow_overlap=True

                      if flow_overlap:
                          # check if already accounted
                          if edge2 in potentially_redundant_edges2 and \
                             edge1 in potentially_redundant_edges2[edge2]:
                              continue

                          if edge1 not in potentially_redundant_edges2:
                            potentially_redundant_edges2[edge1]=[]

                          if edge2 not in potentially_redundant_edges2[edge1]:
                            potentially_redundant_edges2[edge1].append(edge2)

        # determine metapath sources and targets
        processed=[]
        for edge1, value in potentially_redundant_edges1.iteritems():
            source_set = edge1.invertex.difference(self.propositions_set)
            for edge2 in potentially_redundant_edges1[edge1]:
               source_set = source_set.union(edge2.invertex.difference(self.propositions_set))

            for edge3, value in potentially_redundant_edges2.iteritems():
                if len(potentially_redundant_edges2[edge3])< 1:
                    continue
                target_set = edge3.outvertex

                # check if edges apply to overlapping flows
                flow_overlap=False
                prot_propositions1 = self.get_protocol_propositions(edge1.invertex)
                prot_propositions3 = self.get_protocol_propositions(edge3.invertex)
                prot_int = prot_propositions1.intersection(prot_propositions3)

                if (prot_int is not None and len(prot_int)>0):
                  # protocols overlap
                  if 'protocol=1' not in prot_int:
                     # TCP or UDP flow -check port overlap
                     dport_propositions1 = self.get_port_propositions(edge1.invertex)
                     dport_propositions3 = self.get_port_propositions(edge3.invertex)
                     sport_propositions1 = self.get_sport_propositions(edge1.invertex)
                     sport_propositions3 = self.get_sport_propositions(edge3.invertex)
                     dport_int = dport_propositions1.intersection(dport_propositions3)
                     sport_int = sport_propositions1.intersection(sport_propositions3)
                     if (dport_int is not None and len(dport_int)>0) and (sport_int is not None and len(sport_int)>0):
                        # port overlap
                        flow_overlap=True
                  else:
                     flow_overlap=True

                if flow_overlap:
                   if (str(list(source_set)), str(list(target_set)))  in processed:
                       continue
                   processed.append((str(list(source_set)), str(list(target_set))))
                   mps= self.get_all_metapaths_from(source_set, target_set, True)
                   if mps is not None and len(mps)>0:
                       for mp in mps:
                          if mp not in result:
                             result.append(mp)

        #print('returning result..')
        if len(result)==0:
            #print('NO redundancies found1')
            return None

        #print('redundancies found')
        redundancies = result
        if redundancies is not None and len(redundancies)>0:
            actual_redundancies=[]
            for mp in redundancies:
                redundant_edges=[]
                if self.has_redundancies2(mp, redundant_edges=redundant_edges):
                   actual_redundancies += redundant_edges

        filtered= MetagraphHelper().remove_duplicate_redundancies(actual_redundancies)
        if len(filtered)==0:
            #print('NO redundancies found')
            #count=1
            #for mp in result:
            #   print("Policy error %s"%count)
            #   for edge in mp.edge_list:
            #       print "%s"%(str(edge))
            #   count+=1
            #print('NO redundancies found2: END')
            return None

        if True:
            count=1
            for redundancy in filtered:
               print("Policy redundancy %s"%count)
               print "edge0- %s"%(str(redundancy[0]))
               print "edge1- %s"%(str(redundancy[1]))
               count+=1

        return filtered


    def classify_redundancy(self, edge_list):
        classifications=[]
        for edge1 in edge_list:
            for edge2 in edge_list:
                if edge1!=edge2:
                    if int(edge1.label) > int(edge2.label):
                        rule1= PolicyRule(edge1)
                        rule2= PolicyRule(edge2)
                    else:
                        rule1= PolicyRule(edge2)
                        rule2= PolicyRule(edge1)

                    if self.IsShadowedRule(rule1, rule2):
                        desc = 'edge-%s shadowed by edge-%s'%(rule1.Id,rule2.Id)
                        if desc not in classifications:
                            classifications.append(desc) #('edge-%s shadowed by edge-%s'%(edge1.label,edge2.label))

                    elif self.IsGeneralisation(rule1, rule2):
                        desc = 'edge-%s is a generalisation of edge-%s'%(rule1.Id,rule2.Id)
                        if desc not in classifications:
                            classifications.append(desc)

                    elif self.IsPartialOverlap(rule1, rule2):
                        desc = 'edge-%s is a partial overlap of edge-%s'%(rule1.Id,rule2.Id)
                        if desc not in classifications:
                            classifications.append()

        return classifications

    def has_redundancies(self, metapath):
        return (not self.is_dominant_metapath(metapath))

    def print_original_policy(self, edge):
        try:
            source = edge.invertex.difference(edge.attributes)
            #cons = Constraint.query.get(edge_label)
            dest = edge.outvertex
            protocol = self.get_protocols(edge.attributes)
            action = self.get_actions(edge.attributes)
            dports = self.get_tcp_ports(edge.attributes,True)
            if dports is None or len(dports)==0:
                dports = self.get_udp_ports(edge.attributes, True)

            sports = self.get_tcp_ports(edge.attributes, False)
            if sports is None or len(sports)==0:
                sports = self.get_udp_ports(edge.attributes, False)


            #TODO
            attack_sig = None
            log_event = None
            notification = None
            priority = None
            profile_log = None
            profile_malware = None
            profile_spyware = None
            profile_url_filter = None

            print('policy:: source=%s, dest=%s, protocol=%s, sport=%s, dport=%s, action=%s'%
                  (source, dest, protocol, sports, dports, action))

            #print('policy:: source=%s, dest=%s, protocol=%s, sport=%s, dport=%s, action=%s, priority=%s, malware_profile=%s, '
            #      'spyware_profile=%s, url_filter=%s, log_profile=%s, attack_sig=%s, log_event=%s, notification=%s'%
            #      (source, dest, protocol, sports, dports, action, priority, profile_malware,
            #       profile_spyware, profile_url_filter, profile_log, attack_sig, log_event, notification))


        except BaseException, e:
            print('Error::print_original_policy: %s'%str(e))


    def __repr__(self):
        edge_desc = [repr(edge) for edge in self.edges]
        full_desc = ''
        for desc in edge_desc:
            if full_desc == '':
                full_desc = desc
            else:
                full_desc += ', ' + desc
        desc = '%s(%s)' % (str(type(self)), full_desc)
        desc = desc.replace('\\', '')
        return desc

# noinspection PyShadowingNames,PyShadowingNames
@singleton
class MetagraphHelper:
    """ Helper class that facilitates metagraph operations.
    """

    def __init__(self):
        pass

    def add_adjacency_matrices(self, adjacency_matrix1, generator_set1, adjacency_matrix2, generator_set2):
        """ Adds the two adjacency matrices provided and returns a combined matrix.
        :param adjacency_matrix1: numpy.matrix
        :param generator_set1: set
        :param adjacency_matrix2: numpy.matrix
        :param generator_set2: set
        :return: numpy.matrix
        """

        if adjacency_matrix1 is None:
            raise MetagraphException('adjacency_matrix1', resources['value_null'])
        if adjacency_matrix2 is None:
            raise MetagraphException('adjacency_matrix2', resources['value_null'])

        if generator_set1 is None:
            raise MetagraphException('generator_set1', resources['value_null'])
        if generator_set2 is None:
            raise MetagraphException('generator_set2', resources['value_null'])

        # check if the generating sets of the matrices overlap (otherwise no sense in combining metagraphs)
        intersection = generator_set1.intersection(generator_set2)
        if intersection is None:
            raise MetagraphException('generator_sets', resources['no_overlap'])

        #combined_adjacency_matrix = None
        if len(generator_set1.difference(generator_set2)) == 0 and len(generator_set2.difference(generator_set1)) == 0:
            # generating sets are identical..use adjacency matrices as is
            size = len(generator_set1)
            combined_adjacency_matrix = MetagraphHelper().get_null_matrix(size, size)
            for i in range(size):
                for j in range(size):
                    # take the union
                    if adjacency_matrix1[i][j] is None:
                        combined_adjacency_matrix[i][j] = adjacency_matrix2[i][j]
                    elif adjacency_matrix2[i][j] is None:
                        combined_adjacency_matrix[i][j] = adjacency_matrix1[i][j]
                    else:
                        temp = list()
                        temp.append(adjacency_matrix1[i][j])
                        temp.append(adjacency_matrix2[i][j])
                        combined_adjacency_matrix[i][j] = temp

        else:
            # generating sets overlap but are different...need to redefine adjacency matrices before adding them
            combined_generating_set = generator_set1.union(generator_set2)
            mg1 = Metagraph(combined_generating_set)

            # add all metagraph1 edges
            edge_list1 = self.get_edges_in_matrix(adjacency_matrix1, generator_set1)
            for edge in edge_list1:
                mg1.add_edge(edge)
            modified_adjacency_matrix1 = mg1.adjacency_matrix().tolist()

            mg2 = Metagraph(combined_generating_set)
            # add all metagraph2 edges
            edge_list2 = self.get_edges_in_matrix(adjacency_matrix2, generator_set2)
            for edge in edge_list2:
                mg2.add_edge(edge)
            modified_adjacency_matrix2 = mg2.adjacency_matrix().tolist()

            #combined_mg = Metagraph(combined_generating_set)
            size = len(combined_generating_set)
            combined_adjacency_matrix = MetagraphHelper().get_null_matrix(size, size)
            for i in range(size):
                for j in range(size):
                    # take the union
                    if modified_adjacency_matrix1[i][j] is None:
                        combined_adjacency_matrix[i][j] = modified_adjacency_matrix2[i][j]
                    elif modified_adjacency_matrix2[i][j] is None:
                        combined_adjacency_matrix[i][j] = modified_adjacency_matrix1[i][j]
                    else:
                        temp = modified_adjacency_matrix1[i][j]
                        for triple in modified_adjacency_matrix2[i][j]:
                            if not triple in modified_adjacency_matrix1[i][j]:
                                temp.append(triple)
                        combined_adjacency_matrix[i][j] = temp

        return combined_adjacency_matrix

    def get_triples(self, nested_triples_list):
        """ Returns a list of non-nested Triple objects.
        :param nested_triples_list: list of nested Triple objects
        :return: list of Triple objects
        """

        if nested_triples_list is None or len(nested_triples_list) == 0:
            raise MetagraphException('triples_list', resources['value_null'])

        result = []
        if isinstance(nested_triples_list, list):
            for elt in nested_triples_list:
                if isinstance(elt, Triple):
                    result.append(elt)
                else:
                    temp = self.get_triples(elt)
                    for item in temp:
                        result.append(item)

        return result

    def forms_cover(self,triples_set, target, x_j):
        cumulative_output = {x_j}
        for triple in triples_set:
            # retain cooutputs
            cumulative_output = cumulative_output.union(triple.cooutputs)

        return target.issubset(cumulative_output)

    def get_metapath_from_triples(self, source, target, triples_set):
        edges=set()
        for triple in triples_set:
            if isinstance(triple.edges, Edge):
                edges = edges.union({triple.edges})
            else:
                edges = edges.union(triple.edges)

        return Metapath(source,target,edges)

    def multiply_adjacency_matrices(self, adjacency_matrix1, generator_set1, adjacency_matrix2, generator_set2):
        """ Multiplies the two adjacency matrices provided and returns the result.
        :param adjacency_matrix1: numpy.matrix
        :param generator_set1: set
        :param adjacency_matrix2: numpy.matrix
        :param generator_set2: set
        :return: numpy.matrix
        """

        if adjacency_matrix1 is None:
            raise MetagraphException('adjacency_matrix1', resources['value_null'])
        if adjacency_matrix2 is None:
            raise MetagraphException('adjacency_matrix2', resources['value_null'])

        if generator_set1 is None:
            raise MetagraphException('generator_set1', resources['value_null'])
        if generator_set2 is None:
            raise MetagraphException('generator_set2', resources['value_null'])

        # check generating sets are identical
        if not(len(generator_set1.difference(generator_set2)) == 0 and
               len(generator_set2.difference(generator_set1)) == 0):
            raise MetagraphException('generator_sets', resources['not_identical'])

        size = len(generator_set1)
        resultant_adjacency_matrix = MetagraphHelper().get_null_matrix(size, size)

        # O(N3C2)
        for i in range(size):
            for j in range(size):
                # O(NC2)
                resultant_adjacency_matrix[i][j] = self.multiply_components(adjacency_matrix1,
                                                                            adjacency_matrix2,
                                                                            generator_set1, i,
                                                                            j, size)
                #print('multiply_components')

        return resultant_adjacency_matrix

    def multiply_components(self, adjacency_matrix1, adjacency_matrix2, generator_set1, i, j, size):
        """ Multiplies elements of two adjacency matrices.
        :param adjacency_matrix1: numpy.matrix
        :param adjacency_matrix2: numpy.matrix
        :param generator_set1: set
        :param i: int
        :param j: int
        :param size: int
        :return: list of Triple objects.
        """

        if adjacency_matrix1 is None:
            raise MetagraphException('adjacency_matrix1', resources['value_null'])
        if adjacency_matrix2 is None:
            raise MetagraphException('adjacency_matrix2', resources['value_null'])
        if generator_set1 is None or len(generator_set1) == 0:
            raise MetagraphException('generator_set1', resources['value_null'])

        result = []
        # computes the outermost loop (ie., k=1...K where K is the size of each input matrix)
        for k in range(size):
            a_ik = adjacency_matrix1[i][k]
            b_kj = adjacency_matrix2[k][j]
            #print('multiply_triple_lists')
            temp = self.multiply_triple_lists(a_ik, b_kj, list(generator_set1)[i],
                                              list(generator_set1)[j], list(generator_set1)[k])
            if temp is not None:
                #print('len(temp): %s'%len(temp))
                for triple in temp:
                    if not MetagraphHelper().is_triple_in_list(triple, result):
                        result.append(triple)
                    #if triple not in result: result.append(triple)
        if len(result) == 0:
            return None

        return result

    def multiply_components(self, adjacency_matrix1, adjacency_matrix2, generator_set1, i, j, size):
        """ Multiplies elements of two adjacency matrices.
        :param adjacency_matrix1: numpy.matrix
        :param adjacency_matrix2: numpy.matrix
        :param generator_set1: set
        :param i: int
        :param j: int
        :param size: int
        :return: list of Triple objects.
        """

        if adjacency_matrix1 is None:
            raise MetagraphException('adjacency_matrix1', resources['value_null'])
        if adjacency_matrix2 is None:
            raise MetagraphException('adjacency_matrix2', resources['value_null'])
        if generator_set1 is None or len(generator_set1) == 0:
            raise MetagraphException('generator_set1', resources['value_null'])

        result = []
        # computes the outermost loop (ie., k=1...K where K is the size of each input matrix)
        for k in range(size):
            a_ik = adjacency_matrix1[i][k]
            b_kj = adjacency_matrix2[k][j]
            temp = self.multiply_triple_lists(a_ik, b_kj, list(generator_set1)[i],
                                              list(generator_set1)[j], list(generator_set1)[k])

            if temp is not None and len(temp)>0:
                result+=temp

        if len(result) == 0:
            return None

        return list(set(result))

    def multiply_triple_lists(self, triple_list1, triple_list2, x_i, x_j, x_k):
        """ Multiplies two list of Triple objects and returns the result.
        :param triple_list1: list of Triple objects
        :param triple_list2: list of Triple objects
        :param x_i: generator set element
        :param x_j: generator set element
        :param x_k: generator set element
        :return: list of Triple objects
        """

        if triple_list1 is None or triple_list2 is None:
            return None

        triples_list=[]
        # computes the middle loop (ie., n=1...N where N is the size of triple_list1
        for triple1 in triple_list1:
            # computes the innermost loop (ie., m=1...M where M is the size of triple_list2
            for triple2 in triple_list2:
                temp = self.multiply_triples(triple1, triple2, x_i, x_j, x_k)
                if temp is not None:
                    triples_list.append(temp)

        if len(triples_list)>0:
            return list(set(triples_list))

        return []

    @staticmethod
    def multiply_triples(triple1, triple2, x_i, x_j, x_k):
        """ Multiplies two Triple objects and returns the result.
        :param triple1: Triple object
        :param triple2: Triple object
        :param x_i: generator set element
        :param x_j: generator set element
        :param x_k: generator set element
        :return: Triple object
        """

        if triple1 is None or triple2 is None:
            return None

        # compute alpha(R)
        alpha_r = triple2.coinputs
        if triple2.coinputs is None:
            alpha_r = triple1.coinputs
        elif triple1.coinputs is not None:
            alpha_r = triple1.coinputs.union(triple2.coinputs)
        if alpha_r is not None and triple1.cooutputs is not None:
            alpha_r = alpha_r.difference(({x_i}).union(triple1.cooutputs))
        elif alpha_r is not None:
            alpha_r = alpha_r.difference(({x_i}))

        # compute beta(R)
        beta_r = triple2.cooutputs
        if triple2.cooutputs is None:
            beta_r = triple1.cooutputs
        elif triple1.cooutputs is not None:
            beta_r = triple1.cooutputs.union(triple2.cooutputs)
        if beta_r is not None:
            beta_r = beta_r.union({x_k})
            beta_r = beta_r.difference({x_j})
        else:
            beta_r = {x_k}
            beta_r = beta_r.difference(({x_j}))

        # compute gamma(R)
        truncated = []
        if triple1.edges not in truncated:
            if isinstance(triple1.edges, Edge):
                truncated.append(triple1.edges)
            else:
                if isinstance(triple1.edges, list):
                    truncated = copy.copy(triple1.edges)

        if triple2.edges not in truncated:
            if isinstance(triple2.edges, Edge):
                truncated.append(triple2.edges)
            else:
                truncated.append = copy.copy(triple2.edges)

        gamma_r = truncated

        return Triple(alpha_r, beta_r, gamma_r)

    @staticmethod
    def get_null_matrix(rows, cols):
        """ Returns a null matrix of dimension rows x cols.
        :param rows: int
        :param cols: int
        :return: list
        """
        psi = None
        result = []
        for i in range(rows):
            # noinspection PyUnusedLocal
            item = [psi for j in range(cols)]
            result.append(item)
        return result

    @staticmethod
    def get_edges_in_matrix(adjacency_matrix, generator_set):
        """ Returns the list of edges in the provided adjacency matrix.
        :param adjacency_matrix: numpy.matrix
        :param generator_set: set
        :return: list of Edge objects
        """

        if adjacency_matrix is None:
            raise MetagraphException('adjacency_matrix', resources['value_null'])
        if generator_set is None or len(generator_set) == 0:
            raise MetagraphException('generator_set', resources['value_null'])

        size = len(generator_set)
        edge_list = []
        for i in range(size):
            for j in range(size):
                if adjacency_matrix[i][j] is not None:
                    # list of triples
                    triples_list = adjacency_matrix[i][j]
                    for triple in triples_list:
                        # gamma_R describes the edge
                        if triple[2] not in edge_list:
                            edge_list.append(triple[2])

        return edge_list

    def get_edge_list(self, nested_edges):
        """ Returns a non-nested list of edges.
        :param nested_edges: nested list of Edge objects
        :return: non-nested list of Edge objects.
        """

        edge_list = []
        if nested_edges is None or len(nested_edges) == 0:
            return edge_list

        for element in nested_edges:
            if isinstance(element, list):
                temp = self.get_edge_list(element)
                for edge in temp:
                    if edge not in edge_list:
                        edge_list.append(edge)
            elif isinstance(element, Edge):
                if element not in edge_list:
                    edge_list.append(element)

        return edge_list

    def remove_edge_list(self, edges_to_remove, target_edge_list):
        """ Removes a given set of edges from a target set.
        :param edges_to_remove: a list of edges to remove
        :param target_edge_list: a list of target edges
        :return: a list of Edge objects.
        """
        updated = []
        for edge in target_edge_list:
            if not self.is_edge_in_list(edge, edges_to_remove):
                updated.append(edge)

        return updated

    def get_edges_from_triple_list(self, nested_triples):
        """ Returns the edges present in a nested list of Triple objects.
        :param nested_triples: nested list of Triple objects
        :return: non-nested list of Triple objects
        """

        result = []
        if nested_triples is None:
            return result

        if isinstance(nested_triples, Triple):
            return nested_triples.edges

        elif isinstance(nested_triples, list):
            for element in nested_triples:
                temp = self.get_edges_from_triple_list(element)
                if isinstance(temp, list):
                    for elt in temp:
                        if isinstance(elt, Edge) and (elt not in result):
                            result.append(elt)
                        elif isinstance(elt, list):
                            for elt2 in elt:
                                if elt2 not in result:
                                    result.append(elt2)
                elif isinstance(temp, Edge) and (temp not in result):
                    result.append(temp)

        return result

    def is_triple_in_list(self, triple, triples_list):
        """ Checks whether a particular Triple object is in a given list of Triples.
        :param triple: Triple object
        :param triples_list: list of Triple object
        :return: boolean
        """

        if triple==None:
            raise MetagraphException('triple', resources['value_null'])
        if triples_list==None:
            raise MetagraphException('triples_list', resources['value_null'])

        result=False

        for element in triples_list:
            if self.are_triples_equal(triple, element):
               result= True
               break

        return result

    def is_edge_in_list(self, edge, nested_edges):
        """ Checks whether a particular edge is in the nested list of edges.
        :param edge: Edge object
        :param nested_edges: nested list of Edge objects
        :return: boolean
        """

        if edge is None:
            raise MetagraphException('edge', resources['value_null'])
        if nested_edges is None:
            raise MetagraphException('nested_edges', resources['value_null'])
        result = False

        if isinstance(nested_edges, list):
            for element in nested_edges:
                result = self.is_edge_in_list(edge, element)
                if result:
                    break

        elif isinstance(nested_edges, Edge):
            if self.are_edges_equal(edge, nested_edges):
                result = True

        return result

    def is_node_in_list(self, node, node_list):
        """ Checks if a particular node is in the given list of nodes.
        :param node: Node object
        :param node_list: list of Node objects
        :return: boolean
        """

        if node is None:
            raise MetagraphException('node', resources['value_null'])
        if node_list is None:
            raise MetagraphException('node_list', resources['value_null'])
        result = False

        if isinstance(node_list, list):
            for element in node_list:
                result = self.is_node_in_list(node, element)
                if result:
                    break

        elif isinstance(node_list, Node):
            if self.are_nodes_equal(node, node_list):
                result = True

        return result

    def are_triples_equal(self, triple1, triple2):
        """ Checks if the two given triples are equal.
        :param triple1: Triple object
        :param triple2: Triple object
        :return: boolean
        """

        if triple1 is None:
            raise MetagraphException('triple1', resources['value_null'])
        if triple2 is None:
            raise MetagraphException('triple2', resources['value_null'])
        if not isinstance(triple1, Triple):
            raise MetagraphException('triple1', resources['format_invalid'])
        if not isinstance(triple2, Triple):
            raise MetagraphException('triple2', resources['format_invalid'])

        edge_list_match = True
        for edge in triple1.edges:
            if not self.is_edge_in_list(edge, triple2.edges):
                edge_list_match = False
                break

        for edge in triple2.edges:
            if not self.is_edge_in_list(edge, triple1.edges):
                edge_list_match = False
                break

        return (triple1.coinputs == triple2.coinputs and
                triple1.cooutputs == triple2.cooutputs and
                len(triple1.edges) == len(triple2.edges) and
                edge_list_match)

    @staticmethod
    def are_edges_equal(edge1, edge2):
        """ Checks if the two given edges are equal.
        :param edge1: Edge object
        :param edge2: Edge object
        :return: boolean
        """

        if edge1 is None:
            raise MetagraphException('edge1', resources['value_null'])
        if edge2 is None:
            raise MetagraphException('edge2', resources['value_null'])
        if not isinstance(edge1, Edge):
            raise MetagraphException('edge1', resources['format_invalid'])
        if not isinstance(edge2, Edge):
            raise MetagraphException('edge2', resources['format_invalid'])

        if edge1.attributes is not None and edge2.attributes is not None:
            return (edge1.invertex == edge2.invertex and
                    edge1.outvertex == edge2.outvertex and
                    set(edge1.attributes) == set(edge2.attributes) and
                    edge1.label == edge2.label)
        else:
            return (edge1.invertex == edge2.invertex and
                    edge1.outvertex == edge2.outvertex and
                    edge1.label == edge2.label)

    @staticmethod
    def are_nodes_equal(node1, node2):
        """ Checks if two given nodes are equal.
        :param node1: Node object
        :param node2: Node object
        :return: boolean
        """

        if node1 is None:
            raise MetagraphException('node1', resources['value_null'])
        if node2 is None:
            raise MetagraphException('node2', resources['value_null'])
        if not isinstance(node1, Node):
            raise MetagraphException('node1', resources['format_invalid'])
        if not isinstance(node2, Node):
            raise MetagraphException('node2', resources['format_invalid'])

        return node1.element_set == node2.element_set

    @staticmethod
    def is_edge_list_included_recursive(edges, reference_edge_list):
        """ Checks if an edge list is included in the reference edge list.
        :param edges: list of Edge objects
        :param reference_edge_list: reference lists of Edge objects
        :return: boolean
        """

        if edges is None or len(edges) == 0:
            raise MetagraphException('edges', resources['value_null'])
        if reference_edge_list is None or len(reference_edge_list) == 0:
            raise MetagraphException('reference_edge_list', resources['value_null'])

        for ref_edges in reference_edge_list:
            inclusive_list = True
            for edge1 in edges:
                if not MetagraphHelper().is_edge_in_list(edge1, ref_edges):
                #match=False
                #for edge2 in ref_edges:
                #    if edge1.invertex==edge2.invertex and edge1.outvertex==edge2.outvertex:
                #        match=True
                #        break
                #if not match:
                    inclusive_list = False
                    break
            if inclusive_list and len(edges) == len(ref_edges):
                return True

        return False

    @staticmethod
    def is_edge_list_included(edge_list1, edge_list2):
        """ Checks if an edge list is included in edge_list2.
        :param edge_list1: first  list of Edge objects
        :param edge_list2: second list of Edge objects
        :return: boolean
        """

        if edge_list1 is None or len(edge_list1) == 0:
            raise MetagraphException('edge_list1', resources['value_null'])
        if edge_list2 is None or len(edge_list2) == 0:
            raise MetagraphException('edge_list2', resources['value_null'])

        return set(edge_list1).issubset(set(edge_list2))


    @staticmethod
    def is_metapath_included(metapath, metapath_list):
        """ Checks if a metapath is included in a list of metapaths.
        :param metapath: metapath object
        :param metapath_list: metapath list
        :return: boolean
        """
        if metapath_list is None or len(metapath_list) == 0:
            raise MetagraphException('metapath_list', resources['value_null'])
        if metapath is None or len(metapath.edge_list) == 0:
            raise MetagraphException('metapath', resources['value_null'])

        for mp in metapath_list:
            if set(metapath.edge_list).issubset(set(mp.edge_list)):
                return True

            if metapath.source.issubset(mp.source) and \
               metapath.target.issubset(mp.target) and \
               MetagraphHelper().is_attributes_subset(mp.edge_list, metapath.edge_list):
                print('metapath included in mp::')
                print('  metapath=%s,'%(metapath.edge_list))
                print('  mp=%s')%(mp.edge_list)
                #filepath='/Users/a1070571/Documents/ITS/inclusion.dot'
                #edge_list = []
                #edge_list = mp.edge_list + metapath.edge_list
                #MetagraphHelper().generate_visualisation(edge_list,filepath)
                #print('test')
                return True

        return False

    @staticmethod
    def is_attributes_subset(edge_list1, edge_list2):
        """ Checks if the attributes of edge_list1 is included
        in the attributes of the edge_list2.
        :param edge_list1: first edge list
        :param edge_list2: second edge list
        :return: boolean
        """
        # TODO: how should this logic work for >1 edge?
        attr_list1=[]
        attr_list2=[]
        for edge in edge_list1:
            attr_list1 += edge.attributes
        for edge in edge_list2:
            attr_list2 += edge.attributes

        return set(attr_list1).issubset(set(attr_list2))


    def get_netinputs(self, edge_list):
        """ Retrieves a list of net inputs corresponding to the given edge list.
        :param edge_list: list of Edge objects
        :return: list
        """

        if edge_list is None or len(edge_list) == 0:
            raise MetagraphException('edge_list', resources['value_null'])

        all_inputs = []
        for edge in edge_list:
            if isinstance(edge, Edge):
                for input_elt in edge.invertex:
                    if input_elt not in all_inputs:
                        all_inputs.append(input_elt)
            elif isinstance(edge, list):
                temp = self.get_netinputs(edge)
                for item in temp:
                    if item not in all_inputs:
                        all_inputs.append(item)

        return all_inputs

    def get_netoutputs(self, edge_list):
        """ Retrieves a list of net outputs corresponding to the given edge list.
        :param edge_list: list of Edge objects
        :return: list
        """

        if edge_list is None or len(edge_list) == 0:
            raise MetagraphException('edge_list', resources['value_null'])

        all_outputs = []
        for edge in edge_list:
            if isinstance(edge, Edge):
                for output in edge.outvertex:
                    if output not in all_outputs:
                        all_outputs.append(output)
            elif isinstance(edge, list):
                temp = self.get_netoutputs(edge)
                for item in temp:
                    if item not in all_outputs:
                        all_outputs.append(item)

        return all_outputs

    @staticmethod
    def get_coinputs_from_triples(triples_list):
        """ Retrieves a list of co-inputs corresponding to the given triples list.
        :param triples_list: list of Triple objects
        :return: list
        """

        if triples_list is None or len(triples_list) == 0:
            raise MetagraphException('triples_list', resources['value_null'])

        all_coinputs = []
        for triple in triples_list:
            if triple.coinputs is not None:
                for coinput in triple.coinputs:
                    if coinput not in all_coinputs:
                        all_coinputs.append(coinput)

        return all_coinputs

    @staticmethod
    def get_cooutputs_from_triples(triples_list):
        """ Retrieves a list of co-outputs corresponding to the given triples list.
        :param triples_list: list of Triple objects
        :return: list
        """

        if triples_list is None or len(triples_list) == 0:
            raise MetagraphException('triples_list', resources['value_null'])

        all_cooutputs = []
        for triple in triples_list:
            if triple.cooutputs is not None:
                for cooutput in triple.cooutputs:
                    if cooutput not in all_cooutputs:
                        all_cooutputs.append(cooutput)

        return all_cooutputs

    def extract_edge_list(self, nested_edge_list):
        """ Retrieves a non-nested edge list from the given nested list.
        :param nested_edge_list: nested list of Edge objects
        :return: non-nested list of Edge objects.
        """

        if nested_edge_list is None or len(nested_edge_list) == 0:
            raise MetagraphException('nested_edge_list', resources['value_null'])
        result = []

        for element in nested_edge_list:
            if isinstance(element, list):
                temp = self.extract_edge_list(element)
                for item in temp:
                    result.append(item)

            elif isinstance(element, Edge):
                result.append(element)

        return result

    def node_lists_overlap(self, nodes_list1, nodes_list2):
        """ Checks if two lists of nodes overlap.
        :param nodes_list1: list of Node objects
        :param nodes_list2: list of Node objects
        :return: boolean
        """

        for node1 in nodes_list1:
            if self.is_node_in_list(node1, nodes_list2):
                return True

        return False

    def generate_visualisation(self, edge_list, file_path, enable_ranking=False,highlight_nodes=False,highlight_edges=False,nodes_subset=None, resize=False):
        try:
            clusters = dict()
            edges = []
            index = 0
            gen_set = set()
            for edge in edge_list:
                inv=None
                outv=None
                temp_invertex = edge.invertex
                if edge.attributes is not None and len(edge.attributes) >0:
                   temp_invertex = edge.invertex.difference(edge.attributes)

                gen_set = gen_set.union(temp_invertex)
                if len(list(temp_invertex))>1:
                    # is a cluster
                    if temp_invertex not in clusters.values():
                        # create new
                        clusters[index] = temp_invertex
                        inv = index
                        index+=1
                    else:
                        # use existing
                        inv = clusters.values().index(temp_invertex)

                else:
                    # indiv node
                    inv = list(temp_invertex)[0]
                    inv = inv.strip()
                    inv = inv.replace(' ','_')
                    inv = inv.replace('-','_')
                    inv = inv.replace('"','')
                    inv = inv.replace(';','')

                if len(list(edge.outvertex))>1:
                    # a cluster
                    if edge.outvertex not in clusters.values():
                        # create new
                        clusters[index] = edge.outvertex
                        gen_set = gen_set.union(edge.outvertex)
                        outv=index
                        index+=1
                    else:
                        # use existing
                        outv = clusters.values().index(edge.outvertex)

                else:
                    # indiv node
                    gen_set = gen_set.union(edge.outvertex)
                    outv = list(edge.outvertex)[0]
                    outv = outv.strip()
                    outv = outv.replace(' ','_')
                    outv = outv.replace('-','_')
                    outv = outv.replace('"','')
                    outv = outv.replace(';','')

                if (inv, outv, edge.attributes) not in edges:
                    edges.append((inv, outv, edge.attributes))

            #TODO: update overlapping cluster elts
            clusters = self.update_clusters(clusters)

            dot_output=[]
            dot_output.append('digraph G { \n')
            if resize:
                elts = len(list(gen_set))
                if elts <5:
                    dot_output.append('size="8,3"; \n')
                elif elts <10:
                    dot_output.append('size="12,4"; \n')
                else:
                    dot_output.append('size="30,10"; \n')
                dot_output.append('ratio=fill; \n')

            dot_output.append('compound=true; \n')

            elt_tracker=dict()

            # clusters first
            for index, content in clusters.iteritems():
                dot_output.append('subgraph cluster%s { \n'%index)
                for elt in list(content):
                    elt = elt.strip()
                    original=None
                    if '_' in elt:
                        original = elt.replace('_','')
                        original = original.replace(' ','_')

                    elt = elt.replace(' ','_')
                    elt = elt.replace('-','_')
                    elt = elt.replace('"','')
                    elt = elt.replace(';','')
                    if original is not None:
                       if elt not in elt_tracker:
                           elt_tracker[elt]=original
                       elt = elt + ' [label=%s]'%original

                    dot_output.append('%s; \n'%elt)
                dot_output.append('} \n')

            # add edges for tracking duplicate cluster elts
            for key,value in elt_tracker.iteritems():
                dot_output.append('%s -> %s [dir=none color="blue"]; \n'%(value,key))

            if highlight_nodes and nodes_subset is None:
                for elt in list(gen_set):
                    temp = elt.replace(' ','_')
                    dot_output.append('%s [color="blue"];'%temp)

            if highlight_nodes and nodes_subset is not None:
                for elt in list(nodes_subset):
                    temp = elt.replace(' ','_')
                    dot_output.append('%s [color="darkorchid"];'%temp) #darkorchid turquoise

            color = 'black'
            if highlight_edges:
                color='red'

            # add all edges
            for edge in edges:
                inv = None
                outv=None
                inv_cluster=None
                outv_cluster=None
                if isinstance(edge[0],int):
                    # invertex is a cluster
                    inv_cluster='cluster%s'%edge[0]
                    inv = clusters[edge[0]]
                else:
                    # invertex is an individual node
                    inv = edge[0]

                if isinstance(edge[1],int):
                    # outvertex is a cluster
                    outv_cluster='cluster%s'%edge[0]
                    outv = clusters[edge[1]]
                else:
                    # outvertex is an individual node
                    outv = edge[1]

                attributes = None
                attrs = edge[2]
                if attrs is not None and len(attrs)>0:
                    attributes = ','.join(attrs)

                if isinstance(inv,set) and isinstance(outv,set):
                    # inv, outv both clusters
                    a = list(inv)[0]
                    b = list(outv)[0]
                    a = a.strip()
                    a = a.replace(' ','_')
                    a = a.replace('-','_')
                    a = a.replace('"','')
                    a = a.replace(';','')
                    b = b.strip()
                    b = b.replace(' ','_')
                    b = b.replace('-','_')
                    b = b.replace('"','')
                    b = b.replace(';','')
                    if attributes is not None:
                        dot_output.append('%s -> %s [ltail=%s,lhead=%s,label="[%s]",color="%s"]; \n'%(a,b,inv_cluster,outv_cluster,attributes,color))
                    else:
                        dot_output.append('%s -> %s [ltail=%s,lhead=%s,color="%s"]; \n'%(a,b,inv_cluster,outv_cluster,color))

                elif isinstance(inv,set) and isinstance(outv,str):
                    # inv is cluster, outv is string
                    a = list(inv)[0]
                    a = a.strip()
                    a = a.replace(' ','_')
                    a = a.replace('-','_')
                    a = a.replace('"','')
                    a = a.replace(';','')
                    if attributes is not None:
                        dot_output.append('%s -> %s [ltail=%s,label="[%s]",color="%s"]; \n'%(a,outv,inv_cluster,attributes,color))
                    else:
                        dot_output.append('%s -> %s [ltail=%s,color="%s"]; \n'%(a,outv,inv_cluster,color))

                elif isinstance(inv,str) and isinstance(outv,set):
                    # inv is string, outv is cluster
                    b = list(outv)[0]
                    b = b.strip()
                    b = b.replace(' ','_')
                    b = b.replace('-','_')
                    b = b.replace('"','')
                    b = b.replace(';','')
                    if attributes is not None:
                        dot_output.append('%s -> %s [lhead=%s,label="[%s]",color="%s"]; \n'%(inv,b,outv_cluster,attributes,color))
                    else:
                        dot_output.append('%s -> %s [lhead=%s,color="%s"]; \n'%(inv,b,outv_cluster,color))

                else:
                    # inv, outv both string
                    if attributes is not None:
                        dot_output.append('%s -> %s [label="[%s]",color="%s"]; \n'%(inv,outv,attributes,color))
                    else:
                        dot_output.append('%s -> %s [color="%s"]; \n'%(inv,outv,color))

            # enable ranking
            if enable_ranking:
                course_groups = self.get_course_groups(list(gen_set),clusters, MetagraphHelper().lookup_table)
                if course_groups is not None:
                   for key, group in course_groups.iteritems():
                       group_str=''
                       for elt in group:
                           group_str = group_str + '"%s"; '%elt
                       dot_output.append('{ rank = same; %s }'% group_str)

            # write output to .dot file
            dot_output.append('} \n')
            dot_file_text=''
            for line in dot_output:
                dot_file_text +=line

            #write policy file
            dot_file=open(file_path,'w')
            dot_file.write(dot_file_text)
            dot_file.close()

            return clusters

        except BaseException, e:
            print('generate_visualisation:: Error- %s'%e)

        return None

    def generate_visualisation2(self, edge_list, file_path, display_attributes=True, use_temp_label=False, use_edge_label=False):
        try:
            clusters = dict()
            edges = []
            index = 0
            for edge in edge_list:
                inv=None
                outv=None
                label=edge.label
                temp_invertex = edge.invertex
                if edge.attributes is not None and len(edge.attributes) >0:
                   temp_invertex = edge.invertex.difference(edge.attributes)

                if len(list(temp_invertex))>1:
                    # is a cluster
                    if temp_invertex not in clusters.values():
                        # create new
                        clusters[index] = temp_invertex
                        inv = index
                        index+=1
                    else:
                        # use existing
                        inv = clusters.values().index(temp_invertex)

                else:
                    # indiv node
                    inv = str(list(temp_invertex)[0])
                    inv = inv.strip()
                    inv = inv.replace(' ','_')
                    inv = inv.replace('"','')
                    inv = inv.replace(';','')
                    # TODO fix
                    # inv = inv.replace('-','_')
                    if '-' in inv:
                        items=inv.split('-')
                        if items[0]==items[1]:
                            inv="%s"%items[0]
                        else:
                            inv="%s-%s"%(items[0],items[1])

                if len(list(edge.outvertex))>1:
                    # a cluster
                    if edge.outvertex not in clusters.values():
                        # create new
                        clusters[index] = edge.outvertex
                        outv=index
                        index+=1
                    else:
                        # use existing
                        outv = clusters.values().index(edge.outvertex)

                else:
                    # indiv node
                    outv = str(list(edge.outvertex)[0])
                    outv = outv.strip()
                    outv = outv.replace(' ','_')
                    outv = outv.replace('"','')
                    outv = outv.replace(';','')
                    #TODO fix
                    #outv = outv.replace('-','_')
                    if '-' in outv:
                        items=outv.split('-')
                        if items[0]==items[1]:
                           outv="%s"%items[0]
                        else:
                           outv="%s-%s"%(items[0],items[1])


                if (inv, outv, edge.attributes, label) not in edges:
                    edges.append((inv, outv, edge.attributes, label))

            dot_output=[]
            dot_output.append('digraph G { \n')
            dot_output.append('compound=true; \n')
            dot_output.append('size="14,5"; \n')
            dot_output.append('ratio=fill; \n')

            # clusters first
            for index, content in clusters.iteritems():
                dot_output.append('subgraph cluster%s { \n'%index)
                for elt in list(content):
                    #if '2202' in elt:
                    #    print('test')

                    elt = elt.strip()
                    elt = elt.replace(' ','_')
                    elt = elt.replace('"','')
                    elt = elt.replace(';','')
                    if '-' in elt:
                        items=elt.split('-')
                        if items[0]==items[1]:
                           elt="%s"%items[0]
                        else:
                           elt="%s_%s"%(items[0],items[1])

                    dot_output.append('%s; \n'%elt)
                dot_output.append('} \n')

            # add all edges
            index=1
            label_map=dict()
            for edge in edges:
                inv = None
                outv=None
                inv_cluster=None
                outv_cluster=None
                temp_label='e%s'%index
                edge_label=edge[3]
                index+=1
                if isinstance(edge[0],int):
                    # invertex is a cluster
                    inv_cluster='cluster%s'%edge[0]
                    inv = clusters[edge[0]]
                else:
                    # invertex is an individual node
                    inv = edge[0]

                if isinstance(edge[1],int):
                    # outvertex is a cluster
                    outv_cluster='cluster%s'%edge[1]
                    outv = clusters[edge[1]]
                else:
                    # outvertex is an individual node
                    outv = edge[1]

                attributes = None
                attrs = edge[2]
                if attrs is not None and len(attrs)>0:
                    attributes = ','.join(attrs)

                label_map[temp_label]=attributes

                if isinstance(inv,set) and isinstance(outv,set):
                    # inv, outv both clusters
                    a = list(inv)[0]
                    b = list(outv)[0]
                    a = a.strip()
                    a = a.replace(' ','_')
                    #a = a.replace('-','_')
                    a = a.replace('"','')
                    a = a.replace(';','')

                    if '-' in a:
                        items=a.split('-')
                        if items[0]==items[1]:
                           a="%s"%items[0]
                        else:
                           a="%s_%s"%(items[0],items[1])

                    b = b.strip()
                    b = b.replace(' ','_')
                    b = b.replace('-','_')
                    b = b.replace('"','')
                    b = b.replace(';','')

                    if '-' in b:
                        items=b.split('-')
                        if items[0]==items[1]:
                           b="%s"%items[0]
                        else:
                           b="%s_%s"%(items[0],items[1])

                    if display_attributes and attributes is not None:
                        dot_output.append('%s -> %s [ltail=%s,lhead=%s,label="[%s]"]; \n'%(a,b,inv_cluster,outv_cluster,attributes))
                    elif use_temp_label:
                        dot_output.append('%s -> %s [ltail=%s,lhead=%s,label="[%s]"]; \n'%(a,b,inv_cluster,outv_cluster,temp_label))
                    elif use_edge_label:
                        dot_output.append('%s -> %s [ltail=%s,lhead=%s,label="[%s]"]; \n'%(a,b,inv_cluster,outv_cluster,edge_label))
                    else:
                        dot_output.append('%s -> %s [ltail=%s,lhead="[%s]"]; \n'%(a,b,inv_cluster,outv_cluster))

                elif isinstance(inv,set) and isinstance(outv,str):
                    # inv is cluster, outv is string
                    a = list(inv)[0]
                    a = a.strip()
                    a = a.replace(' ','_')
                    #a = a.replace('-','_')
                    a = a.replace('"','')
                    a = a.replace(';','')

                    if '-' in a:
                        items=a.split('-')
                        if items[0]==items[1]:
                           a="%s"%items[0]
                        else:
                           a="%s_%s"%(items[0],items[1])

                    if display_attributes and attributes is not None:
                        dot_output.append('%s -> %s [ltail=%s,label="[%s]"]; \n'%(a,outv,inv_cluster,attributes))
                    elif use_temp_label:
                        dot_output.append('%s -> %s [ltail=%s,label="[%s]"]; \n'%(a,outv,inv_cluster,temp_label))
                    elif use_edge_label:
                        dot_output.append('%s -> %s [ltail=%s,label="[%s]"]; \n'%(a,outv,inv_cluster,edge_label))
                    else:
                        dot_output.append('%s -> %s [ltail="[%s]"]; \n'%(a,outv,inv_cluster))

                elif isinstance(inv,str) and isinstance(outv,set):
                    # inv is string, outv is cluster
                    b = list(outv)[0]
                    b = b.strip()
                    b = b.replace(' ','_')
                    #b = b.replace('-','_')
                    b = b.replace('"','')
                    b = b.replace(';','')

                    if '-' in b:
                        items=b.split('-')
                        if items[0]==items[1]:
                           b="%s"%items[0]
                        else:
                           b="%s_%s"%(items[0],items[1])

                    if display_attributes and attributes is not None:
                        dot_output.append('%s -> %s [lhead=%s,label="[%s]"]; \n'%(inv,b,outv_cluster,attributes))
                    elif use_temp_label:
                        dot_output.append('%s -> %s [lhead=%s,label="[%s]"]; \n'%(inv,b,outv_cluster,temp_label))
                    elif use_edge_label:
                        dot_output.append('%s -> %s [lhead=%s,label="[%s]"]; \n'%(inv,b,outv_cluster,edge_label))
                    else:
                        dot_output.append('%s -> %s [lhead="[%s]"]; \n'%(inv,b,outv_cluster))

                else:
                    # inv, outv both string
                    if display_attributes and attributes is not None:
                        dot_output.append('%s -> %s [label="[%s]"]; \n'%(inv,outv,attributes))
                    elif use_temp_label:
                        dot_output.append('%s -> %s [label="[%s]"]; \n'%(inv,outv,temp_label))
                    elif use_edge_label:
                        dot_output.append('%s -> %s [label="[%s]"]; \n'%(inv,outv,edge_label))
                    else:
                        dot_output.append('%s -> %s; \n'%(inv,outv))


            # write output to .dot file
            dot_output.append('} \n')
            dot_file_text=''
            for line in dot_output:
                dot_file_text +=line

            #write policy file
            dot_file=open(file_path,'w')
            dot_file.write(dot_file_text)
            dot_file.close()

            # write service list
            services_text = ''
            for key, services in label_map.iteritems():
                services_text = services_text + '%s: %s \n'%(key,str(services))

            import os
            folder = os.path.dirname(file_path)
            filename = os.path.join(folder,'services.txt')
            svc_file=open(filename,'w')
            svc_file.write(services_text)
            svc_file.close()

            #print(label_map)

        except BaseException, e:
            print('generate_visualisation:: Error- %s'%e)

    def generate_visualisation1(self, edge_list, file_path):
        try:
            clusters = dict()
            edges = []
            index = 0
            for edge in edge_list:
                inv=None
                outv=None

                if len(list(edge.invertex))>1:
                    # is a cluster
                    if edge.invertex not in clusters.values():
                        # create new
                        clusters[index] = edge.invertex
                        inv = index
                        index+=1
                    else:
                        # use existing
                        inv = clusters.values().index(edge.invertex)

                else:
                    # indiv node
                    inv = list(edge.invertex)[0]
                    inv = inv.strip()
                    inv = inv.replace(' ','_')
                    inv = inv.replace('-','_')
                    inv = inv.replace('"','')
                    inv = inv.replace(';','')

                if len(list(edge.outvertex))>1:
                    # a cluster
                    if edge.outvertex not in clusters.values():
                        # create new
                        clusters[index] = edge.outvertex
                        outv=index
                        index+=1
                    else:
                        # use existing
                        outv = clusters.values().index(edge.outvertex)

                else:
                    # indiv node
                    outv = list(edge.outvertex)[0]
                    outv = outv.strip()
                    outv = outv.replace(' ','_')
                    outv = outv.replace('-','_')
                    outv = outv.replace('"','')
                    outv = outv.replace(';','')

                if (inv, outv) not in edges:
                    edges.append((inv, outv))

            dot_output=[]
            dot_output.append('digraph G { \n')
            dot_output.append('compound=true; \n')

            # clusters first
            for index, content in clusters.iteritems():
                dot_output.append('subgraph cluster%s { \n'%index)
                for elt in list(content):
                    #if '2202' in elt:
                    #    print('test')

                    elt = elt.strip()
                    elt = elt.replace(' ','_')
                    elt = elt.replace('-','_')
                    elt = elt.replace('"','')
                    elt = elt.replace(';','')
                    dot_output.append('%s; \n'%elt)
                dot_output.append('} \n')

            # add all edges
            for edge in edges:
                inv = None
                outv=None
                inv_cluster=None
                outv_cluster=None
                if isinstance(edge[0],int):
                    # invertex is a cluster
                    inv_cluster='cluster%s'%edge[0]
                    inv = clusters[edge[0]]
                else:
                    # invertex is an individual node
                    inv = edge[0]

                if isinstance(edge[1],int):
                    # outvertex is a cluster
                    outv_cluster='cluster%s'%edge[0]
                    outv = clusters[edge[1]]
                else:
                    # outvertex is an individual node
                    outv = edge[1]

                if isinstance(inv,set) and isinstance(outv,set):
                    # inv, outv both clusters
                    a = list(inv)[0]
                    b = list(outv)[0]
                    a = a.strip()
                    a = a.replace(' ','_')
                    a = a.replace('-','_')
                    a = a.replace('"','')
                    a = a.replace(';','')
                    b = b.strip()
                    b = b.replace(' ','_')
                    b = b.replace('-','_')
                    b = b.replace('"','')
                    b = b.replace(';','')
                    dot_output.append('%s -> %s [ltail=%s,lhead=%s]; \n'%(a,b,inv_cluster,outv_cluster))
                elif isinstance(inv,set) and isinstance(outv,str):
                    # inv is cluster, outv is string
                    a = list(inv)[0]
                    a = a.strip()
                    a = a.replace(' ','_')
                    a = a.replace('-','_')
                    a = a.replace('"','')
                    a = a.replace(';','')
                    dot_output.append('%s -> %s [ltail=%s]; \n'%(a,outv,inv_cluster))
                elif isinstance(inv,str) and isinstance(outv,set):
                    # inv is string, outv is cluster
                    b = list(outv)[0]
                    b = b.strip()
                    b = b.replace(' ','_')
                    b = b.replace('-','_')
                    b = b.replace('"','')
                    b = b.replace(';','')
                    dot_output.append('%s -> %s [lhead=%s]; \n'%(inv,b,outv_cluster))
                else:
                    # inv, outv both string
                    dot_output.append('%s -> %s; \n'%(inv,outv))

            # write output to .dot file
            dot_output.append('} \n')
            dot_file_text=''
            for line in dot_output:
                dot_file_text +=line

            #write policy file
            dot_file=open(file_path,'w')
            dot_file.write(dot_file_text)
            dot_file.close()

        except BaseException, e:
            print('generate_visualisation:: Error- %s'%e)

    @staticmethod
    def update_clusters(clusters):
        cluster_overlaps=dict()
        for index, cluster in clusters.iteritems():
            for elt in list(cluster):
                if elt not in cluster_overlaps:
                    cluster_overlaps[elt]=[]
                if index not in cluster_overlaps[elt]:
                    cluster_overlaps[elt].append(index)

        for elt, cluster_indices in cluster_overlaps.iteritems():
            if len(cluster_indices)>1:
                # duplicate element
                sorted_indices = sorted(cluster_indices)
                # mark primary
                primary = sorted_indices[0]
                # mark secondaries
                updated_elt=elt
                for index in sorted_indices[1:]:
                    clusters[index].remove(elt)
                    updated_elt = updated_elt + '_'
                    updated = list(clusters[index])
                    updated.append(updated_elt)
                    clusters[index] = set(updated)

        return clusters

    @staticmethod
    def nodes_overlap(nodes_list1, nodes_list2):
        """ Checks if two lists of nodes overlap.
        :param nodes_list1: list of Node objects
        :param nodes_list2: list of Node objects
        :return: boolean
        """

        for node1 in nodes_list1:
            for node2 in nodes_list2:
                intersection = node1.get_element_set().intersection(node2.get_element_set())
                if intersection is not None and len(intersection) > 0:
                    return True

        return False

    @staticmethod
    def get_generating_set(edge_list):
        """ Retrieves the generating set of the metagraph from its edge list.
        :param edge_list: list of Edge objects
        :return: set
        """

        if edge_list is None or len(edge_list) == 0:
            raise MetagraphException('edge_list', resources['value_null'])

        generating_set = []
        for edge in edge_list:
            for input_elt in edge.invertex:
                if input_elt not in generating_set:
                    generating_set.append(input_elt)

            for output in edge.outvertex:
                if output not in generating_set:
                    generating_set.append(output)

        return set(generating_set)

    @staticmethod
    def get_element_set(nodes_list):
        """ Retrieves the set of elements within a given list of nodes
        :param nodes_list: list of Node objects
        :return: set
        """
        if nodes_list is not None and len(nodes_list) > 0:
            result = set()
            for node in nodes_list:
                result = result.union(node.get_element_set())

            return result

        return set()

    @staticmethod
    def transpose_matrix(matrix):
        """ Computes the transpose matrix of given matrix
        :param matrix: 2D array
        :return: 2D array
        """

        if matrix is None:
            raise MetagraphException('matrix', resources['value_null'])

        #rows = len(matrix)
        cols = len(matrix[0])

        result = []

        for j in range(cols):
            column = [row[j] for row in matrix]
            result.append(column)

        return result

    @staticmethod
    def custom_multiply_matrices(matrix1, matrix2, edge_list):
        """Multiplies the Triple lists of two matrices
        :param matrix1: 2D array
        :param matrix2: 2D array
        :param edge_list: list of Edge objects
        :return: 2D array
        """

        if matrix1 is None:
            raise MetagraphException('matrix1', resources['value_null'])
        if matrix2 is None:
            raise MetagraphException('matrix2', resources['value_null'])
        if edge_list is None or len(edge_list) == 0:
            raise MetagraphException('edge_list', resources['value_null'])

        matrix1_cols = len(matrix1[0])
        matrix2_rows = len(matrix2)
        if matrix1_cols != matrix2_rows:
            raise MetagraphException('matrix1, matrix2', resources['structures_incompatible'])

        result = MetagraphHelper().get_null_matrix(len(matrix1), len(matrix2[0]))

        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                intermediate_result = set()
                for k in range(len(matrix1[0])):
                    a_ik = matrix1[i][k]
                    b_kj = matrix2[k][j]
                    temp = MetagraphHelper().custom_add_matrix_elements(k, a_ik, b_kj, edge_list)
                    intermediate_result = intermediate_result.union(temp)

                result[i][j] = intermediate_result

        return result

    @staticmethod
    def custom_add_matrix_elements(k, a_ik, b_kj, y):
        """Custom addition of matrix elements.
        :param k: int
        :param a_ik: int
        :param b_kj: int
        :param y: list
        :return: set
        """

        if len(y) < k+1:
            raise MetagraphException('k', resources['value_out_of_bounds'])

        if a_ik == 1 and b_kj == -1:
            return {(1, y[k])}
        elif a_ik == -1 and b_kj == -1:
            return {(-1, y[k])}
        else:
            return set()

    @staticmethod
    def extract_edge_label_components(label):
        """Extracts components of an edge label.
        :param label: string
        :return: string tuple
        """

        if label is None or label == '':
            raise MetagraphException('label', resources['value_null'])

        label = label.replace('>', '')
        items = label.split('<')
        if len(items) < 2:
            raise MetagraphException('label', resources['format_invalid'])

        r_ij = {items[0]}
        tuples = items[1].split(';')
        if len(tuples) < 2:
            raise MetagraphException('label', resources['format_invalid'])

        t_a = {tuples[0]}
        t_b = {tuples[1]}

        # noinspection PyRedundantParentheses
        return (r_ij, t_a, t_b)

    @staticmethod
    def get_pre_requisites_list(pre_requisites_desc):
        result=[]
        if pre_requisites_desc=='NA' or pre_requisites_desc=='':
            return result
        items= pre_requisites_desc.split(' or ')
        for item in items:
            item = item.replace('(','')
            item = item.replace(')','')
            sub_items = item.split(' and ')
            if len(sub_items)>1 and sub_items not in result:
                result.append(sub_items)
            elif item not in result:
                result.append(item)

        return result

    @staticmethod
    def CreateLineGraph(multi_digraph):
        import networkx as nx
        #multi_digraph = MetagraphHelper().GetMultiDigraph(conditional_metagraph)

        if multi_digraph:
            line_graph_edges = []
            for v in multi_digraph.nodes():
                incoming=[]
                outgoing=[]
                for edge1 in multi_digraph.in_edges(nbunch=[v]):
                    incoming.append(edge1)
                for edge2 in multi_digraph.out_edges(nbunch=[v]):
                    outgoing.append(edge2)
                line_graph_edges.extend((e1, e2) for e1 in incoming for e2 in outgoing)

            line_graph = nx.DiGraph()
            line_graph.add_edges_from(line_graph_edges)
            line_graph_nodes=[]
            for source,dest in line_graph.nodes():
                # get corresponding attribute
                node_label = multi_digraph.get_edge_data(source,dest)[0]['label']
                line_graph.node[(source,dest)]['label']=node_label
                if node_label not in line_graph_nodes:
                    line_graph_nodes.append(node_label)

            # add disconnected edges from original graph as nodes
            missing=dict()
            for source,dest in multi_digraph.edges():
                matches=multi_digraph.get_edge_data(source,dest)
                for key,val in matches.iteritems():
                    edge_label=val['label']
                    if (edge_label not in line_graph_nodes):
                        if ((source,dest) not in missing):
                            missing[(source,dest)]= []
                        if edge_label not in missing[(source,dest)]:
                           missing[(source,dest)].append(edge_label)

            # node_index=len(line_graph_nodes)+1
            for key,val in missing.iteritems():
                line_graph.add_node(key)
                line_graph.node[key]['label']=val

            return line_graph

        return None


    @staticmethod
    def print_duplicate_edges(edge_list):
        temp = []
        duplicates=[]
        for edge in edge_list:
            if edge in temp and edge not in duplicates:
                duplicates.append(edge)
            else:
                temp.append(edge)

        count=1
        for edge in duplicates:
            print('%s. duplicate edge: %s'%(count,str(edge)))
            count+=1

    @staticmethod
    def GetMultiDigraph(conditional_metagraph):
        import networkx as nx
        multi_digraph= nx.MultiDiGraph()
        # add nodes
        temp=[]
        elts=[]
        index=1
        #for node in conditional_metagraph.nodes:
        #    multi_digraph.add_node(node)
        #    elts.append(list(node.element_set))
        #    temp.append(node)
        #..and edges
        for edge in conditional_metagraph.edges:
            inv=edge.invertex.difference(edge.attributes)
            outv=edge.outvertex.difference(edge.attributes)
            for inv_elt in inv:
                for outv_elt in outv:
                    inv_node=Node({inv_elt})
                    outv_node=Node({outv_elt})
                    if inv_elt not in elts:
                        elts.append(inv_elt)
                        temp.append(inv_node)
                    else:
                        # use existing
                        inv_node= MetagraphHelper().GetNode({inv_elt},temp)
                    if outv_elt not in elts:
                        elts.append(outv_elt)
                        temp.append(outv_node)
                    else:
                        # use existing
                        outv_node= MetagraphHelper().GetNode({outv_elt},temp)

                    #label='e%s'%index
                    multi_digraph.add_edge(inv_node,outv_node,attributes=edge.attributes,label=edge.label)
                    index+=1

        return multi_digraph

    @staticmethod
    def GetNode(vertex_elts, nodes_list):
        for node in nodes_list:
            if node.element_set==vertex_elts: return node
        return None

    @staticmethod
    def is_policy_consistent(original_cmg, surpress_message_output= False, conflict_resolution_scheme=None):
        try:
            nw_rule_count=0

            # convert original metagraph to an intermediate metagraph that can be analysed
            # reformat attributes
            # TCP
            tcp_port_attributes = MetagraphHelper().filter_port_attributes("TCP", original_cmg.propositions_set)
            non_overlapping_tcp_port_ranges = MetagraphHelper().get_minimal_non_overlapping_port_ranges('TCP','dport',tcp_port_attributes)
            # UDP
            udp_port_attributes = MetagraphHelper().filter_port_attributes("UDP", original_cmg.propositions_set)
            non_overlapping_udp_port_ranges = MetagraphHelper().get_minimal_non_overlapping_port_ranges('UDP','dport',udp_port_attributes)

            timestamp_attributes=[]
            for edge in original_cmg.edges:
                # timestamp
                start = MetagraphHelper().filter_timestamp_attributes("start", edge.attributes)
                end = MetagraphHelper().filter_timestamp_attributes("end", edge.attributes)
                if len(start)>0 and len(end)>0:
                    # convert 24 hour timestamps to mins
                    time_range = '%s-%s'%(start[0].replace('start=',''), end[0].replace('end=',''))
                    time_range_mins = MetagraphHelper().get_timestamp_minutes(time_range)
                    if time_range_mins not in timestamp_attributes:
                        timestamp_attributes.append(time_range_mins)

            non_overlapping_timestamp_ranges = MetagraphHelper().get_minimal_non_overlapping_timestamp_ranges(timestamp_attributes)

            propositions_set=original_cmg.propositions_set
            #numeric_ipadresses =  MetagraphHelper().get_ipaddresses_numeric(original_cmg.variables_set)
            #non_overlapping_ipaddress_ranges = MetagraphHelper().get_minimal_non_overlapping_ipaddress_ranges(numeric_ipadresses)

            #converted=[]
            #for ipaddr_range in non_overlapping_ipaddress_ranges:
            #    range_string = '%s-%s'%(ipaddr_range[0],ipaddr_range[1])
            #    converted.append(range_string)

            variable_set = original_cmg.variables_set #set(converted)

            new_edge_list=[]
            for edge in original_cmg.edges:
                edge_attributes= copy.copy(edge.attributes)
                # convert tcp port attributes to non-overlapping ranges via lookup
                invertex = edge.invertex.difference(edge_attributes)
                outvertex = edge.outvertex
                #invertex = MetagraphHelper().get_ipaddresses_numeric(invertex)
                #outvertex = MetagraphHelper().get_ipaddresses_numeric(outvertex)
                #invertex = MetagraphHelper().get_ipaddresses_canonical_form(invertex, non_overlapping_ipaddress_ranges)
                #outvertex = MetagraphHelper().get_ipaddresses_canonical_form(outvertex, non_overlapping_ipaddress_ranges)

                temp= edge.attributes
                # TCP
                edge_tcp_attr= MetagraphHelper().filter_port_attributes("TCP", edge_attributes)
                if len(edge_tcp_attr)>0:
                    canonical_tcp_attr= MetagraphHelper().get_ports_canonical_form('TCP.dport=', edge_tcp_attr[0], non_overlapping_tcp_port_ranges)
                # UDP
                edge_udp_attr= MetagraphHelper().filter_port_attributes("UDP", edge_attributes)
                if len(edge_udp_attr)>0:
                    canonical_udp_attr= MetagraphHelper().get_ports_canonical_form('UDP.dport=', edge_udp_attr[0], non_overlapping_udp_port_ranges)

                # timestamp
                timestamp_start= MetagraphHelper().filter_timestamp_attributes("start", edge_attributes)
                timestamp_end = MetagraphHelper().filter_timestamp_attributes("end", edge_attributes)
                if len(timestamp_start)>0 and len(timestamp_end)>0:
                    # convert 24 hour timestamps to mins
                    time_range = '%s-%s'%(timestamp_start[0].replace('start=',''), timestamp_end[0].replace('end=',''))
                    edge_time_attr = MetagraphHelper().get_timestamp_minutes(time_range)
                    canonical_time_attr= MetagraphHelper().get_timestamp_canonical_form(edge_time_attr, non_overlapping_timestamp_ranges)

                # update temp and propositions_set
                # TCP
                if len(edge_tcp_attr)>0:
                    if edge_tcp_attr[0] in temp: temp.remove(edge_tcp_attr[0])
                    if edge_tcp_attr[0] in propositions_set: propositions_set.remove(edge_tcp_attr[0])
                    for canonical_attr in canonical_tcp_attr:
                        if canonical_attr not in temp:
                            temp.append(canonical_attr)
                        if canonical_attr not in propositions_set:
                            propositions_set= propositions_set.union({canonical_attr})
                # UDP
                if len(edge_udp_attr)>0:
                    if edge_udp_attr[0] in temp: temp.remove(edge_udp_attr[0])
                    if edge_udp_attr[0] in propositions_set: propositions_set.remove(edge_udp_attr[0])
                    for canonical_attr in canonical_udp_attr:
                        if canonical_attr not in temp:
                            temp.append(canonical_attr)
                        if canonical_attr not in propositions_set:
                            propositions_set= propositions_set.union({canonical_attr})

                # time
                if len(timestamp_start)>0 and len(timestamp_end)>0:
                    temp.remove(timestamp_start[0])
                    temp.remove(timestamp_end[0])
                    propositions_set= propositions_set.difference({timestamp_start[0]})
                    propositions_set= propositions_set.difference({timestamp_end[0]})
                    for canonical_attr in canonical_time_attr:
                        if canonical_attr not in temp:
                            temp.append(canonical_attr)
                        if canonical_attr not in propositions_set:
                            propositions_set= propositions_set.union({canonical_attr})

                new_edge= Edge(invertex, outvertex, temp, label=edge.label)
                new_edge_list.append(new_edge)

            # create intermediate metagraph suitable for performing analysis
            converted_cmg= ConditionalMetagraph(variable_set, propositions_set)
            converted_cmg.add_edges_from(new_edge_list)

            if False:
                # reduce metagraph complexity by generating an equivalent metagraph
                if not surpress_message_output:
                    print('generating complexity reduced, equivalent metagraph..')
                group_name_index=1
                new_edge_list=[]
                new_var_set=set()
                invertex_element_group_lookup=dict()
                outvertex_element_group_lookup=dict()

                for edge in converted_cmg.edges:
                    invertex_elts = list(edge.invertex.difference(propositions_set))
                    outvertex_elts = list(edge.outvertex.difference(propositions_set))

                    for elt in invertex_elts:
                        result= converted_cmg.get_associated_edges(elt)
                        edge_list_string=repr(result)
                        if edge_list_string not in invertex_element_group_lookup:
                            invertex_element_group_lookup[edge_list_string]=[]
                        if elt not in invertex_element_group_lookup[edge_list_string]:
                            invertex_element_group_lookup[edge_list_string].append(elt)

                    for elt in outvertex_elts:
                        result= converted_cmg.get_associated_edges(elt)
                        edge_list_string=repr(result)
                        if edge_list_string not in outvertex_element_group_lookup:
                            outvertex_element_group_lookup[edge_list_string]=[]
                        if elt not in outvertex_element_group_lookup[edge_list_string]:
                            outvertex_element_group_lookup[edge_list_string].append(elt)

                import json
                group_details_lookup=dict()

                for edge_list_string, elt_list in invertex_element_group_lookup.iteritems():
                    group_elts = set(elt_list)
                    if len(group_elts)>1:
                        # group
                        group_elts_string = json.dumps(list(group_elts))
                        if group_elts_string not in group_details_lookup:
                            group_name= 'group_%s'%group_name_index
                            group_details_lookup[group_elts_string]=group_name
                            group_name_index+=1

                for edge_list_string, elt_list in outvertex_element_group_lookup.iteritems():
                    group_elts = set(elt_list)
                    if len(group_elts)>1:
                        # group
                        group_elts_string = json.dumps(list(group_elts))
                        if group_elts_string not in group_details_lookup:
                            group_name= 'group_%s'%group_name_index
                            group_details_lookup[group_elts_string]=group_name
                            group_name_index+=1

                # replace original invertices with groups
                for edge in converted_cmg.edges:
                    invertex_elts = list(edge.invertex.difference(propositions_set))
                    outvertex_elts = list(edge.outvertex)
                    new_invertex=set()
                    new_outvertex=set()

                    for group_elts, group_name in group_details_lookup.iteritems():
                        group_elts_list = json.loads(group_elts)
                        group_elts_list = [elt2.encode('ascii', errors='backslashreplace') for elt2 in group_elts_list]
                        if set(group_elts_list).issubset(set(invertex_elts)):
                            new_invertex = new_invertex.union({group_name})
                            invertex_elts = list(set(invertex_elts).difference(set(group_elts_list)))

                    for group_elts, group_name in group_details_lookup.iteritems():
                        group_elts_list = json.loads(group_elts)
                        group_elts_list = [elt2.encode('ascii', errors='backslashreplace') for elt2 in group_elts_list]
                        if set(group_elts_list).issubset(set(outvertex_elts)):
                            new_outvertex = new_outvertex.union({group_name})
                            outvertex_elts = list(set(outvertex_elts).difference(set(group_elts_list)))

                    new_invertex=new_invertex.union(set(invertex_elts))
                    new_outvertex=new_outvertex.union(set(outvertex_elts))
                    new_var_set = new_var_set.union(new_invertex)
                    new_var_set = new_var_set.union(new_outvertex)
                    new_edge_list.append(Edge(new_invertex, new_outvertex,
                                                      attributes=list(edge.invertex.intersection(propositions_set)),
                                                      label=edge.label))

                # create complexity reduced, equivalent metagraph
                reduced_cmg= ConditionalMetagraph(new_var_set, propositions_set)
                reduced_cmg.add_edges_from(new_edge_list)

            # temp
            reduced_cmg = converted_cmg

            #print('Converted Metagraph: %s'%type(reduced_cmg))
            #display_metagraph(reduced_cmg)
            conflict=False

            if not surpress_message_output:
               print('generator-set comparison: original=%s ,reduced=%s'%(len(converted_cmg.generating_set), len(reduced_cmg.generating_set)))

            # print('checking for conflicts...')
            potential_conflicts = reduced_cmg.check_conflicts() # check_conflicts check_conflicts_simple
            if potential_conflicts:
                return "Policy is Inconsistent!"
            else:
                return "Policy is Consistent!"

        except BaseException,e:
            print('Error::is_policy_consistent- %s'%str(e))
            return "Policy is Inconsistent!"

    @staticmethod
    def remove_duplicates(metapaths_list):

        result=[]
        # sort the metapaths
        sorted_metapaths=[]
        lookup= dict()
        for metapath in metapaths_list:
            count = len(metapath.edge_list)
            if count not in lookup:
                lookup[count]=[]
            lookup[count].append(metapath)

        keys = sorted(lookup.keys())
        #print('keys= %s'%keys)
        #for key, value in lookup.iteritems():
        #    print('key(# edges)= %s, count(# occurrences)= %s'%(key,len(value)))

        for key in keys:
            for metapath in lookup[key]:
                sorted_metapaths.append(metapath)

        for metapath1 in sorted_metapaths:
            ignore=False
            for metapath2 in result:
                if MetagraphHelper().is_edge_list_included(metapath1.edge_list, metapath2.edge_list):
                # intersection = set(metapath1.edge_list).intersection(set(metapath2.edge_list))
                # if intersection is not None and len(intersection)>1:
                #   conflict_sources=[]
                #   metapath = Metapath(metapath1.source,metapath1.target,intersection)
                #   if reduced_cmg.has_conflicts(metapath, conflict_sources):
                       # already covered..ignore
                       ignore=True
                       #print('ignore')
                       break

            if not ignore and metapath1 not in result:
                result.append(metapath1)

        return result

    @staticmethod
    def remove_duplicate_redundancies(redundancies):

        result=[]
        for edge_pair1 in redundancies:
            ignore=False
            for edge_pair2 in result:
                if MetagraphHelper().is_edge_list_included(list(edge_pair1),list(edge_pair2)):
                     # already covered..ignore
                     ignore=True
                     break

            if not ignore and edge_pair1 not in result:
                result.append(edge_pair1)

        return result

    @staticmethod
    def filter_port_attributes(protocol, all_attributes):
        filtered=[]
        for attribute in all_attributes:
            if (('%s.dport'%(protocol)) in attribute) and (attribute not in filtered):
                filtered.append(attribute)

        return filtered

    @staticmethod
    def filter_timestamp_attributes(prefix, all_attributes):
        filtered=[]
        for attribute in all_attributes:
            if (prefix in attribute) and (attribute not in filtered):
                filtered.append(attribute)

        return filtered

    @staticmethod
    def get_timestamp_minutes(timestamp_hrs_mins):
        converted_timestamp= timestamp_hrs_mins.encode('ascii', errors='backslashreplace')
        items = converted_timestamp.split('-')
        hrs_start = items[0].split(':')[0]
        mins_start = items[0].split(':')[1]
        total_mins_start= int(hrs_start)*60 + int(mins_start)

        hrs_end = items[1].split(':')[0]
        mins_end = items[1].split(':')[1]
        total_mins_end = int(hrs_end)*60 + int(mins_end)

        if total_mins_end< total_mins_start:
           # ends the next day
           total_mins_end = total_mins_end + 24*60

        return '%s-%s'%(total_mins_start,total_mins_end)

    @staticmethod
    def get_ports_canonical_form(attr_descriptor, edge_tcp_attr, non_overlapping_tcp_port_ranges):
        source = None
        result=[]
        edge_tcp_attr = edge_tcp_attr.replace(attr_descriptor,'')
        if '-' in edge_tcp_attr:
            # port range
            converted_port = edge_tcp_attr.encode('ascii', errors='backslashreplace')
            items= converted_port.split('-')
            source = (int(items[0]), int(items[1]))
        else:
            # single value
            converted_port = edge_tcp_attr.encode('ascii', errors='backslashreplace')
            source=(int(converted_port), int(converted_port))

        for non_overlapping_port_range in non_overlapping_tcp_port_ranges:
            if non_overlapping_port_range[0] >= source[0] and \
               non_overlapping_port_range[1] <= source[1]:
                 formatted = '%s%s-%s'%(attr_descriptor,non_overlapping_port_range[0],non_overlapping_port_range[1])
                 result.append(formatted)

        return result

    @staticmethod
    def get_ipaddresses_canonical_form(ipaddress_ranges, non_overlapping_ipaddress_ranges):
        source = None
        result=[]

        for ipaddress_range in ipaddress_ranges:
            if '-' in ipaddress_range:
                items= ipaddress_range.split('-')
                source = (long(items[0]), long(items[1]))
            else:
                converted_address = ipaddress_range.encode('ascii', errors='backslashreplace')
                source=(long(converted_address), long(converted_address))

            for non_overlapping_ipaddress_range in non_overlapping_ipaddress_ranges:
                if non_overlapping_ipaddress_range[0] >= source[0] and \
                   non_overlapping_ipaddress_range[1] <= source[1]:
                     formatted = '%s-%s'%(non_overlapping_ipaddress_range[0], non_overlapping_ipaddress_range[1])
                     result.append(formatted)

        return set(result)

    @staticmethod
    def get_timestamp_canonical_form(edge_time_attr, non_overlapping_time_ranges):
        source = None
        result=[]
        converted_timestamp = edge_time_attr.encode('ascii', errors='backslashreplace')
        if '-' in edge_time_attr:
            # port range
            items= converted_timestamp.split('-')
            source = (int(items[0]), int(items[1]))
        else:
            # single value
            source=(int(converted_timestamp), int(converted_timestamp))

        for non_overlapping_time_range in non_overlapping_time_ranges:
            if non_overlapping_time_range[0] >= source[0] and \
               non_overlapping_time_range[1] <= source[1]:
                 formatted = 'active_time=%s-%s'%(non_overlapping_time_range[0],non_overlapping_time_range[1])
                 result.append(formatted)

        return result

    @staticmethod
    def get_minimal_non_overlapping_port_ranges(protocol, port_type, overlapping_port_ranges):
         desc_tag='%s.%s='%(protocol,port_type)
         temp=[]

         index=1
         for port_or_range in overlapping_port_ranges:
             if desc_tag in port_or_range:
                port_or_range = port_or_range.replace(desc_tag,'')
                if '-' in port_or_range:
                    # range
                    items=port_or_range.split('-')
                    val=(int(items[0]), int(items[1]), str(index))
                    if val not in temp: temp.append(val)
                else:
                    # single port value
                    val = (int(port_or_range), int(port_or_range), str(index))
                    if val not in temp: temp.append(val)
             index+=1

         endpoints = sorted(list(set([r[0] for r in temp] + [r[1] for r in temp])))
         final=[]
         index=0
         last_end=None
         for e in endpoints:
             # look for tuples beginning with e
             matches= MetagraphHelper().get_tuples_starting_with(e,temp)
             # check if point exists
             points= [match for match in matches if match[0]==match[1]]
             if len(points)>0 and ((e,e) not in final):
                 final.append((e,e))
                 last_end=e
             # check if range(s) exist
             ranges= [match for match in matches if match[0]!=match[1]]
             if len(ranges)>0:
                for range in ranges:
                    start=range[0]
                    end=range[1]
                    # get next endpoint value
                    next_val = endpoints[index+1]
                    ref_tuples1= MetagraphHelper().get_tuples_starting_with(next_val,temp)
                    ref_tuples2= MetagraphHelper().get_tuples_ending_with(next_val,temp)
                    if len(ref_tuples1)>0:
                       end=next_val-1
                       if (start<=end) and (start,end) not in final:
                         final.append((start,end))
                    elif len(ref_tuples2)>0:
                       end=next_val
                       if (start<=end) and (start,end) not in final:
                         final.append((start,end))

                    last_end=end

             # look for tuples ending with e
             matches2= MetagraphHelper().get_tuples_ending_with(e,temp)
             if len(matches2)>0 and index < len(endpoints)-1:
                 start=last_end+1
                 next_val = endpoints[index+1]
                 ref_tuples1= MetagraphHelper().get_tuples_starting_with(next_val,temp)
                 ref_tuples2= MetagraphHelper().get_tuples_ending_with(next_val,temp)
                 if len(ref_tuples1)>0:
                    end=next_val-1
                    if (start<=end) and (start,end) not in final:
                      final.append((start,end))
                 elif len(ref_tuples2)>0:
                    end=next_val
                    if (start<=end) and (start,end) not in final:
                      final.append((start,end))

                 last_end=end
             index+=1
         return final

    @staticmethod
    def get_minimal_non_overlapping_timestamp_ranges(overlapping_time_ranges):
         result=[]
         temp=[]

         index=1
         for time_range in overlapping_time_ranges:
            if '-' in time_range:
                items=time_range.split('-')
                val=(int(items[0]), int(items[1]), str(index))
                if val not in temp: temp.append(val)
            index+=1

         endpoints = sorted(list(set([r[0] for r in temp] + [r[1] for r in temp])))
         final=[]
         index=0
         last_end=None
         for e in endpoints:
             # look for tuples beginning with e
             matches= MetagraphHelper().get_tuples_starting_with(e,temp)
             # check if point exists
             points= [match for match in matches if match[0]==match[1]]
             if len(points)>0 and ((e,e) not in final):
                 final.append((e,e))
                 last_end=e
             # check if range(s) exist
             ranges= [match for match in matches if match[0]!=match[1]]
             if len(ranges)>0:
                for range in ranges:
                    start=range[0]
                    end=range[1]
                    # get next endpoint value
                    next_val = endpoints[index+1]
                    ref_tuples1= MetagraphHelper().get_tuples_starting_with(next_val,temp)
                    ref_tuples2= MetagraphHelper().get_tuples_ending_with(next_val,temp)
                    if len(ref_tuples1)>0:
                       end=next_val-1
                       if (start<=end) and (start,end) not in final:
                         final.append((start,end))
                    elif len(ref_tuples2)>0:
                       end=next_val
                       if (start<=end) and (start,end) not in final:
                         final.append((start,end))

                    last_end=end

             # look for tuples ending with e
             matches2= MetagraphHelper().get_tuples_ending_with(e,temp)
             if len(matches2)>0 and index < len(endpoints)-1:
                 start=last_end+1
                 next_val = endpoints[index+1]
                 ref_tuples1= MetagraphHelper().get_tuples_starting_with(next_val,temp)
                 ref_tuples2= MetagraphHelper().get_tuples_ending_with(next_val,temp)
                 if len(ref_tuples1)>0:
                    end=next_val-1
                    if (start<=end) and (start,end) not in final:
                      final.append((start,end))
                 elif len(ref_tuples2)>0:
                    end=next_val
                    if (start<=end) and (start,end) not in final:
                      final.append((start,end))

                 last_end=end
             index+=1
         return final

    @staticmethod
    def get_minimal_non_overlapping_ipaddress_ranges(overlapping_ipaddress_ranges):
         result=[]
         temp=[]

         index=1
         for ipaddress_range in overlapping_ipaddress_ranges:
            if '-' in ipaddress_range:
                items=ipaddress_range.split('-')
                val=(long(items[0]), long(items[1]), str(index))
                if val not in temp: temp.append(val)
            index+=1

         endpoints = sorted(list(set([r[0] for r in temp] + [r[1] for r in temp])))
         final=[]
         index=0
         last_end=None
         for e in endpoints:
             # look for tuples beginning with e
             matches= MetagraphHelper().get_tuples_starting_with(e,temp)
             # check if range(s) exist
             # check if point exists
             points= [match for match in matches if match[0]==match[1]]
             if len(points)>0 and ((e,e) not in final):
                 final.append((e,e))
                 last_end=e
             ranges= [match for match in matches if match[0]!=match[1]]
             if len(ranges)>0:
                for range in ranges:
                    start=range[0]
                    end=range[1]
                    # get next endpoint value
                    next_val = endpoints[index+1]
                    ref_tuples1= MetagraphHelper().get_tuples_starting_with(next_val,temp)
                    ref_tuples2= MetagraphHelper().get_tuples_ending_with(next_val,temp)
                    if len(ref_tuples1)>0:
                       end=next_val-1
                       if (start<=end) and (start,end) not in final:
                         final.append((start,end))
                    elif len(ref_tuples2)>0:
                       end=next_val
                       if (start<=end) and (start,end) not in final:
                         final.append((start,end))
                    last_end=end

             # look for tuples ending with e
             matches2= MetagraphHelper().get_tuples_ending_with(e,temp)
             if len(matches2)>0 and index < len(endpoints)-1:
                 start=last_end+1
                 next_val = endpoints[index+1]
                 ref_tuples1= MetagraphHelper().get_tuples_starting_with(next_val,temp)
                 ref_tuples2= MetagraphHelper().get_tuples_ending_with(next_val,temp)
                 if len(ref_tuples1)>0:
                    end=next_val-1
                    if (start<=end) and (start,end) not in final:
                      final.append((start,end))
                 elif len(ref_tuples2)>0:
                    end=next_val
                    if (start<=end) and (start,end) not in final:
                      final.append((start,end))

                 last_end=end
             index+=1
         return final

    @staticmethod
    def get_tuples_starting_with(start_val,tuples_list):
         return [t for t in tuples_list if t[0]==start_val]

    @staticmethod
    def get_tuples_ending_with(end_val,tuples_list):
         return [t for t in tuples_list if t[1]==end_val]




    #TODO: put these in a separate helper class (MUD related)

    @staticmethod
    def get_device_acl_details(extracted_data,ignore_rules=False):
        from exception import InvalidMUDFileException

        acl_details= dict()

        # TODO: what are these subnets (ayyoob/hassan)?
        local_networks = {'192.168.1.0/24'}
        local_gateway = {'192.168.1.1/32'}

        if extracted_data is not None:
            from_device_acls = extracted_data['ietf-mud:mud']['from-device-policy']['access-lists']['access-list']
            if len(from_device_acls)>0:
                for acl in extracted_data['ietf-mud:mud']['from-device-policy']['access-lists']['access-list']:
                    acl_name = acl['name']
                    # lookup ACL data
                    data = extracted_data['ietf-access-control-list:access-lists']['acl']
                    if len(data)>0:
                        for acl_data in data:
                            if acl_name == acl_data['name']:
                                aces= acl_data['aces']
                                if len(aces)>0:
                                    for ace in aces['ace']:
                                        source='device' # ace['matches']['l3']['ietf-mud:direction-initiated']
                                        dest=set(['0.0.0.0/0'])
                                        substituted=False
                                        if 'ietf-mud:mud' in ace['matches']:
                                            if 'controller' in ace['matches']['ietf-mud:mud']: # local-networks
                                               dest = ace['matches']['ietf-mud:mud']['controller'] # dest.union({"10.10.0.7/32"}) # dest.union({'controllers'}) #["10.10.0.7/32"]     #  ace['matches']['ietf-mud:mud']['controller']
                                               if 'urn:ietf:params:mud:gateway' in dest or \
                                                  'urn:ietf:params:mud:dns' in dest or \
                                                  'urn:ietf:params:mud:ntp' in dest:
                                                     dest = local_gateway
                                                     substituted=True
                                            if 'local-networks' in ace['matches']['ietf-mud:mud']: #internet
                                               dest = ace['matches']['ietf-mud:mud']['local-networks'] #dest.union({"10.10.0.0/24"}) #dest.union(local_networks) #["0.0.0.0/0"]
                                               if len(dest)==1 and dest[0]==None:
                                                   dest = local_networks
                                                   substituted=True
                                            if 'manufacturer' in ace['matches']['ietf-mud:mud']: #internet
                                               dest = ace['matches']['ietf-mud:mud']['manufacturer'] #dest.union({"201.10.0.5/32"}) #dest.union({'manufacturer domains'}) #["0.0.0.0/0"]

                                            if 'model' in ace['matches']['ietf-mud:mud']: #internet
                                               dest = ace['matches']['ietf-mud:mud']['model']#dest.union({"10.10.0.8/32"}) #dest.union({'same-model devices'}) #["0.0.0.0/0"]
                                            if '*' in ace['matches']['ietf-mud:mud']: #internet
                                               dest = ace['matches']['ietf-mud:mud']['local-networks'] #dest.union({"10.10.0.0/24"}) #dest.union(local_networks) #["0.0.0.0/0"]


                                        if 'ipv4' in ace['matches']:
                                            protocol = ace['matches']['ipv4']['protocol']
                                            sports_start=None
                                            sports_end=None
                                            dports_start=None
                                            dports_end=None
                                            action= str(ace['actions']['forwarding']).strip().lower()

                                            if action!='drop' and action!='accept':
                                                raise InvalidMUDFileException("%s : %s"%(resources['mud_file_action_invalid'],action))

                                            if 'ietf-acldns:dst-dnsname' in ace['matches']['ipv4']:
                                               dest = [ace['matches']['ipv4']['ietf-acldns:dst-dnsname']]
                                            if 'destination-ipv4-network' in ace['matches']['ipv4']:
                                               dest = [ace['matches']['ipv4']['destination-ipv4-network']]

                                               if not substituted and CanonicalPolicyHelper().is_private_ipaddress(list(dest)[0]): #.split('/')[0]
                                                   raise InvalidMUDFileException("%s : %s"%(resources['mud_file_has_local_ipaddress_details'],list(dest)[0]))

                                            if 'udp' in ace['matches'] and \
                                                'destination-port' in ace['matches']['udp'] and \
                                                ace['matches']['udp']['destination-port'] is not None:
                                                dports_start= ace['matches']['udp']['destination-port']['port']
                                                dports_end= ace['matches']['udp']['destination-port']['port']
                                            elif 'tcp' in ace['matches'] and \
                                                'destination-port' in ace['matches']['tcp'] and \
                                                ace['matches']['tcp']['destination-port'] is not None:
                                                dports_start= ace['matches']['tcp']['destination-port']['port']
                                                dports_end= ace['matches']['tcp']['destination-port']['port']
                                            elif 'udp' in ace['matches'] and \
                                                'source-port' in ace['matches']['udp'] and \
                                                ace['matches']['udp']['source-port'] is not None:
                                                sports_start= ace['matches']['udp']['source-port']['port']
                                                sports_end= ace['matches']['udp']['source-port']['port']
                                            elif 'tcp' in ace['matches'] and \
                                                'source-port' in ace['matches']['tcp'] and \
                                                ace['matches']['tcp']['source-port'] is not None:
                                                sports_start= ace['matches']['tcp']['source-port']['port']
                                                sports_end= ace['matches']['tcp']['source-port']['port']

                                            if protocol==6 or protocol==17:
                                                sports = (1024,65535) #(0,65535)
                                                dports = (1024,65535) #(0,65535)
                                            else:
                                                sports=None
                                                dports=None

                                            if dports_end==0:
                                                continue

                                            if sports_start is not None and sports_end is not None:
                                                sports=(sports_start,sports_end)
                                            if dports_start is not None and dports_end is not None:
                                                dports=(dports_start,dports_end)

                                            if ignore_rules and protocol==6 and sports!=(1024,65535): #sports!=(0,65535):
                                                continue

                                            extracted_ace = ACE(source,dest,protocol,dports,sports,action)
                                            if 'from' not in acl_details:
                                                acl_details['from']=[]

                                            acl_details['from'].append(extracted_ace)

                                        if 'l2' in ace['matches']:
                                            #TODO: handle
                                            print('L2 tag not yet handled')

            to_device_acls = extracted_data['ietf-mud:mud']['to-device-policy']['access-lists']['access-list']
            if len(to_device_acls)>0:
                for acl in extracted_data['ietf-mud:mud']['to-device-policy']['access-lists']['access-list']:
                    acl_name = acl['name']
                    # lookup ACL data
                    data = extracted_data['ietf-access-control-list:access-lists']['acl']
                    if len(data)>0:
                        for acl_data in data:
                            if acl_name == acl_data['name']:
                                aces= acl_data['aces']
                                if len(aces)>0:
                                    for ace in aces['ace']:
                                        dest='device'#ace['matches']['tcp-acl']['ietf-mud:direction-initiated']
                                        source=set(['0.0.0.0/0'])
                                        substituted=False
                                        if 'ietf-mud:mud' in ace['matches']:
                                            if 'controller' in ace['matches']['ietf-mud:mud']: # local-networks
                                               source = ace['matches']['ietf-mud:mud']['controller'] # dest.union({"10.10.0.7/32"}) # dest.union({'controllers'}) #["10.10.0.7/32"]     #  ace['matches']['ietf-mud:mud']['controller']
                                               if 'urn:ietf:params:mud:gateway' in source or \
                                                  'urn:ietf:params:mud:dns' in source or \
                                                  'urn:ietf:params:mud:ntp' in source:
                                                   source = local_gateway
                                                   substituted=True
                                            if 'local-networks' in ace['matches']['ietf-mud:mud']: #internet
                                               source = ace['matches']['ietf-mud:mud']['local-networks'] #dest.union({"10.10.0.0/24"}) #dest.union(local_networks) #["0.0.0.0/0"]
                                               if len(source)==1 and source[0]==None:
                                                  source = local_networks
                                                  substituted=True
                                            if 'manufacturer' in ace['matches']['ietf-mud:mud']: #internet
                                               source = ace['matches']['ietf-mud:mud']['manufacturer'] #dest.union({"201.10.0.5/32"}) #dest.union({'manufacturer domains'}) #["0.0.0.0/0"]
                                            if 'model' in ace['matches']['ietf-mud:mud']: #internet
                                               source = ace['matches']['ietf-mud:mud']['model']#dest.union({"10.10.0.8/32"}) #dest.union({'same-model devices'}) #["0.0.0.0/0"]

                                            #if 'controller' in ace['matches']['ietf-mud:mud']: # local-networks
                                            #   source = source.union({"10.10.0.7/32"}) # dest.union({'controllers'}) #["10.10.0.7/32"]     #  ace['matches']['ietf-mud:mud']['controller']
                                            #if 'local-networks' in ace['matches']['ietf-mud:mud']: #internet
                                            #   source = source.union({"10.10.0.0/24"}) #dest.union(local_networks) #["0.0.0.0/0"]
                                            #if 'manufacturer' in ace['matches']['ietf-mud:mud']: #internet
                                            #   source = source.union({"201.10.0.5/32"}) #dest.union({'manufacturer domains'}) #["0.0.0.0/0"]
                                            #if 'model' in ace['matches']['ietf-mud:mud']: #internet
                                            #   source = source.union({"10.10.0.8/32"}) #dest.union({'same-model devices'}) #["0.0.0.0/0"]


                                        if 'ipv4' in ace['matches']:
                                            protocol = ace['matches']['ipv4']['protocol']
                                            sports_start=None
                                            sports_end=None
                                            dports_start=None
                                            dports_end=None
                                            action= str(ace['actions']['forwarding']).strip().lower()

                                            if action!='drop' and action!='accept':
                                                raise InvalidMUDFileException("%s : %s"%(resources['mud_file_action_invalid'],action))

                                            if 'ietf-acldns:src-dnsname' in ace['matches']['ipv4']: # local-networks
                                               source = [ace['matches']['ipv4']['ietf-acldns:src-dnsname']] # dest.union({"10.10.0.7/32"}) # dest.union({'controllers'}) #["10.10.0.7/32"]     #  ace['matches']['ietf-mud:mud']['controller']
                                            if 'source-ipv4-network' in ace['matches']['ipv4']: # local-networks
                                               source = [ace['matches']['ipv4']['source-ipv4-network']] # dest.union({"10.10.0.7/32"}) # dest.union({'controllers'}) #["10.10.0.7/32"]     #  ace['matches']['ietf-mud:mud']['controller']

                                            if not substituted and CanonicalPolicyHelper().is_private_ipaddress(list(source)[0]):
                                                   raise InvalidMUDFileException("%s : %s"%(resources['mud_file_has_local_ipaddress_details'], list(source)[0]))

                                            if 'udp' in ace['matches'] and \
                                               'source-port' in ace['matches']['udp'] and \
                                                ace['matches']['udp']['source-port'] is not None:
                                                sports_start= ace['matches']['udp']['source-port']['port']
                                                sports_end= ace['matches']['udp']['source-port']['port']
                                            elif 'tcp' in ace['matches'] and \
                                                'source-port' in ace['matches']['tcp'] and \
                                                ace['matches']['tcp']['source-port'] is not None:
                                                sports_start= ace['matches']['tcp']['source-port']['port']
                                                sports_end= ace['matches']['tcp']['source-port']['port']
                                            elif 'udp' in ace['matches'] and \
                                                'destination-port' in ace['matches']['udp'] and \
                                                ace['matches']['udp']['destination-port'] is not None:
                                                dports_start= ace['matches']['udp']['destination-port']['port']
                                                dports_end= ace['matches']['udp']['destination-port']['port']
                                            elif 'tcp' in ace['matches'] and \
                                                'destination-port' in ace['matches']['tcp'] and \
                                                ace['matches']['tcp']['destination-port'] is not None:
                                                dports_start= ace['matches']['tcp']['destination-port']['port']
                                                dports_end= ace['matches']['tcp']['destination-port']['port']


                                            if protocol==6 or protocol==17:
                                                sports = (1024,65535) #(0,65535)
                                                dports = (1024,65535) #(0,65535)
                                            else:
                                                sports=None
                                                dports=None

                                            if dports_end==0:
                                                continue

                                            if sports_start is not None and sports_end is not None:
                                                sports=(sports_start,sports_end)
                                            if dports_start is not None and dports_end is not None:
                                                dports=(dports_start,dports_end)

                                            if ignore_rules and protocol==6 and sports!=(1024,65535): #sports!=(0,65535):
                                                continue

                                            extracted_ace = ACE(source,dest,protocol,dports,sports,action)
                                            if 'to' not in acl_details:
                                                acl_details['to']=[]

                                            acl_details['to'].append(extracted_ace)

                                        if 'l2' in ace['matches']:
                                            #TODO: handle
                                            print('L2 tag not yet handled')


            #print(extracted['ietf-mud:mud']['from-device-policy']['access-lists']['access-list'][0]['acl-name'])
            #print(extracted['ietf-access-control-list:access-lists']['acl'][0]['access-list-entries']['ace'][0]['matches']['ipv6-acl']['ietf-acldns:src-dnsname'])

            return acl_details

    def get_device_acl_details2(extracted_data):

        acl_details= dict()
        #acl_details = MetagraphHelper().lookup_acl_details_by_direction(extracted_data,'from-device-policy',acl_details)
        #acl_details = MetagraphHelper().lookup_acl_details_by_direction(extracted_data,'to-device-policy',acl_details)
        #return acl_details

        local_networks = {'controllers', 'same-model devices'}

        # format - as used by mudmaker.com
        if extracted_data is not None:

            from_device_acls = extracted_data['ietf-mud:mud']['from-device-policy']['access-lists']['access-list']
            if len(from_device_acls)>0:
                for acl in extracted_data['ietf-mud:mud']['from-device-policy']['access-lists']['access-list']:
                    acl_name = acl['name']
                    # lookup ACL data
                    data = extracted_data['ietf-access-control-list:access-lists']['acl']
                    if len(data)>0:
                        for acl_data in data:
                            if acl_name == acl_data['name']:
                                aces= acl_data['aces']
                                if len(aces)>0:
                                    for ace in aces['ace']:
                                        source='device' # ace['matches']['l3']['ietf-mud:direction-initiated']
                                        dest=set()
                                        if 'controller' in ace['matches']['ietf-mud:mud']: # local-networks
                                           dest = dest.union({"10.10.0.7/32"}) # dest.union({'controllers'}) #["10.10.0.7/32"]     #  ace['matches']['ietf-mud:mud']['controller']
                                        if 'local-networks' in ace['matches']['ietf-mud:mud']: #internet
                                           dest = dest.union({"10.10.0.0/24"}) #dest.union(local_networks) #["0.0.0.0/0"]
                                        if 'manufacturer' in ace['matches']['ietf-mud:mud']: #internet
                                           dest = dest.union({"201.10.0.5/32"}) #dest.union({'manufacturer domains'}) #["0.0.0.0/0"]
                                        if 'model' in ace['matches']['ietf-mud:mud']: #internet
                                           dest = dest.union({"10.10.0.8/32"}) #dest.union({'same-model devices'}) #["0.0.0.0/0"]

                                        protocol = ace['matches']['l3']['ipv4']['protocol']
                                        sports_start=None
                                        sports_end=None
                                        dports_start=None
                                        dports_end=None
                                        action=ace['actions']['forwarding']
                                        if 'l4' in ace['matches'] and 'udp' in ace['matches']['l4'] and \
                                            'destination-port-range-or-operator' in ace['matches']['l4']['udp'] and \
                                            ace['matches']['l4']['udp']['destination-port-range-or-operator'] is not None:
                                            dports_start= ace['matches']['l4']['udp']['destination-port-range-or-operator']['port']
                                            dports_end= ace['matches']['l4']['udp']['destination-port-range-or-operator']['port']
                                        elif 'l4' in ace['matches'] and 'tcp' in ace['matches']['l4'] and \
                                            'destination-port-range-or-operator' in ace['matches']['l4']['tcp'] and \
                                            ace['matches']['l4']['tcp']['destination-port-range-or-operator'] is not None:
                                            dports_start= ace['matches']['l4']['tcp']['destination-port-range-or-operator']['port']
                                            dports_end= ace['matches']['l4']['tcp']['destination-port-range-or-operator']['port']
                                        elif 'l4' in ace['matches'] and 'udp' in ace['matches']['l4'] and \
                                            'source-port' in ace['matches']['l4']['udp'] and \
                                            ace['matches']['l4']['udp']['source-port'] is not None:
                                            sports_start= ace['matches']['l4']['udp']['source-port']['port']
                                            sports_end= ace['matches']['l4']['udp']['source-port']['port']
                                        elif 'l4' in ace['matches'] and 'tcp' in ace['matches']['l4'] and \
                                            'source-port' in ace['matches']['l4']['tcp'] and \
                                            ace['matches']['l4']['tcp']['source-port'] is not None:
                                            sports_start= ace['matches']['l4']['tcp']['source-port']['port']
                                            sports_end= ace['matches']['l4']['tcp']['source-port']['port']

                                        if protocol==6 or protocol==17:
                                            sports = (0,65535)
                                            dports = (0,65535)
                                        else:
                                            sports=None
                                            dports=None

                                        if sports_start is not None and sports_end is not None:
                                            sports=(sports_start,sports_end)
                                        if dports_start is not None and dports_end is not None:
                                            dports=(dports_start,dports_end)

                                        ace = ACE(source,dest,protocol,dports,sports,action)
                                        if 'from' not in acl_details:
                                            acl_details['from']=[]

                                        acl_details['from'].append(ace)


            to_device_acls = extracted_data['ietf-mud:mud']['to-device-policy']['access-lists']['access-list']
            if len(to_device_acls)>0:
                for acl in extracted_data['ietf-mud:mud']['to-device-policy']['access-lists']['access-list']:
                    acl_name = acl['name']
                    # lookup ACL data
                    data = extracted_data['ietf-access-control-list:access-lists']['acl']
                    if len(data)>0:
                        for acl_data in data:
                            if acl_name == acl_data['name']:
                                aces= acl_data['aces']
                                if len(aces)>0:
                                    for ace in aces['ace']:
                                        dest='device'#ace['matches']['tcp-acl']['ietf-mud:direction-initiated']
                                        source=set()
                                        #if 'controller' in ace['matches']['ietf-mud:mud']: # local-networks
                                        #   source = source.union({'controllers'}) #["10.10.0.7/32"]     #  ace['matches']['ietf-mud:mud']['controller']
                                        #if 'local-networks' in ace['matches']['ietf-mud:mud']: #internet
                                        #   source  = source.union(local_networks) #["0.0.0.0/0"]
                                        #if 'manufacturer' in ace['matches']['ietf-mud:mud']: #internet
                                        #   source  = source.union({'manufacturer domains'}) #["0.0.0.0/0"]
                                        #if 'model' in ace['matches']['ietf-mud:mud']: #internet
                                        #   source  = source.union({'same-model devices'}) #["0.0.0.0/0"]

                                        if 'controller' in ace['matches']['ietf-mud:mud']: # local-networks
                                           source = source.union({"10.10.0.7/32"}) # dest.union({'controllers'}) #["10.10.0.7/32"]     #  ace['matches']['ietf-mud:mud']['controller']
                                        if 'local-networks' in ace['matches']['ietf-mud:mud']: #internet
                                           source = source.union({"10.10.0.0/24"}) #dest.union(local_networks) #["0.0.0.0/0"]
                                        if 'manufacturer' in ace['matches']['ietf-mud:mud']: #internet
                                           source = source.union({"201.10.0.5/32"}) #dest.union({'manufacturer domains'}) #["0.0.0.0/0"]
                                        if 'model' in ace['matches']['ietf-mud:mud']: #internet
                                           source = source.union({"10.10.0.8/32"}) #dest.union({'same-model devices'}) #["0.0.0.0/0"]


                                        protocol = ace['matches']['l3']['ipv4']['protocol']
                                        sports_start=None
                                        sports_end=None
                                        dports_start=None
                                        dports_end=None
                                        action=ace['actions']['forwarding']
                                        if 'l4' in ace['matches'] and 'udp' in ace['matches']['l4'] and \
                                           'source-port-range-or-operator' in ace['matches']['l4']['udp'] and \
                                            ace['matches']['l4']['udp']['source-port-range-or-operator'] is not None:
                                            sports_start= ace['matches']['l4']['udp']['source-port-range-or-operator']['port']
                                            sports_end= ace['matches']['l4']['udp']['source-port-range-or-operator']['port']
                                        elif 'l4' in ace['matches'] and 'tcp' in ace['matches']['l4'] and \
                                            'source-port-range-or-operator' in ace['matches']['l4']['tcp'] and \
                                            ace['matches']['l4']['tcp']['source-port-range-or-operator'] is not None:
                                            sports_start= ace['matches']['l4']['tcp']['source-port-range-or-operator']['port']
                                            sports_end= ace['matches']['l4']['tcp']['source-port-range-or-operator']['port']
                                        elif 'l4' in ace['matches'] and 'udp' in ace['matches']['l4'] and \
                                            'destination-port-range-or-operator' in ace['matches']['l4']['udp'] and \
                                            ace['matches']['l4']['udp']['destination-port-range-or-operator'] is not None:
                                            dports_start= ace['matches']['l4']['udp']['destination-port-range-or-operator']['port']
                                            dports_end= ace['matches']['l4']['udp']['destination-port-range-or-operator']['port']
                                        elif 'l4' in ace['matches'] and 'tcp' in ace['matches']['l4'] and \
                                            'destination-port-range-or-operator' in ace['matches']['l4']['tcp'] and \
                                            ace['matches']['l4']['tcp']['destination-port-range-or-operator'] is not None:
                                            dports_start= ace['matches']['l4']['tcp']['destination-port-range-or-operator']['port']
                                            dports_end= ace['matches']['l4']['tcp']['destination-port-range-or-operator']['port']


                                        if protocol==6 or protocol==17:
                                            sports = (0,65535)
                                            dports = (0,65535)
                                        else:
                                            sports=None
                                            dports=None

                                        if sports_start is not None and sports_end is not None:
                                            sports=(sports_start,sports_end)
                                        if dports_start is not None and dports_end is not None:
                                            dports=(dports_start,dports_end)

                                        ace = ACE(source,dest,protocol,dports,sports,action)
                                        if 'to' not in acl_details:
                                            acl_details['to']=[]

                                        acl_details['to'].append(ace)


            #print(extracted['ietf-mud:mud']['from-device-policy']['access-lists']['access-list'][0]['acl-name'])
            #print(extracted['ietf-access-control-list:access-lists']['acl'][0]['access-list-entries']['ace'][0]['matches']['ipv6-acl']['ietf-acldns:src-dnsname'])

            return acl_details

    def get_device_acl_details1(extracted_data):

        acl_details= dict()
        #acl_details = MetagraphHelper().lookup_acl_details_by_direction(extracted_data,'from-device-policy',acl_details)
        #acl_details = MetagraphHelper().lookup_acl_details_by_direction(extracted_data,'to-device-policy',acl_details)
        #return acl_details

        # format as per IETF spec (expires april 2018)
        if extracted_data is not None:

            from_device_acls = extracted_data['ietf-mud:mud']['from-device-policy']['access-lists']['access-list']
            if len(from_device_acls)>0:
                for acl in extracted_data['ietf-mud:mud']['from-device-policy']['access-lists']['access-list']:
                    acl_name = acl['acl-name']
                    # lookup ACL data
                    data = extracted_data['ietf-access-control-list:access-lists']['acl']
                    if len(data)>0:
                        for acl_data in data:
                            if acl_name == acl_data['acl-name']:
                                aces= acl_data['access-list-entries']['ace']
                                if len(aces)>0:
                                    for ace in aces:
                                        source=ace['matches']['tcp-acl']['ietf-mud:direction-initiated']
                                        dest = ace['matches']['ipv6-acl']['ietf-acldns:dst-dnsname']
                                        protocol = ace['matches']['ipv6-acl']['protocol']
                                        sports_start=None
                                        sports_end=None
                                        dports_start=None
                                        dports_end=None
                                        action=ace['actions']['forwarding']
                                        if ace['matches']['ipv6-acl']['destination-port-range'] is not None:
                                            dports_start= ace['matches']['ipv6-acl']['destination-port-range']['lower-port']
                                            dports_end= ace['matches']['ipv6-acl']['destination-port-range']['upper-port']

                                        sports = (0,65535)
                                        dports = (0,65535)
                                        if sports_start is not None and sports_end is not None:
                                            sports=(sports_start,sports_end)
                                        if dports_start is not None and dports_end is not None:
                                            dports=(dports_start,dports_end)

                                        ace = ACE(source,dest,protocol,dports,sports,action)
                                        if 'from' not in acl_details:
                                            acl_details['from']=[]

                                        acl_details['from'].append(ace)


            to_device_acls = extracted_data['ietf-mud:mud']['to-device-policy']['access-lists']['access-list']
            if len(to_device_acls)>0:
                for acl in extracted_data['ietf-mud:mud']['to-device-policy']['access-lists']['access-list']:
                    acl_name = acl['acl-name']
                    # lookup ACL data
                    data = extracted_data['ietf-access-control-list:access-lists']['acl']
                    if len(data)>0:
                        for acl_data in data:
                            if acl_name == acl_data['acl-name']:
                                aces= acl_data['access-list-entries']['ace']
                                if len(aces)>0:
                                    for ace in aces:
                                        dest=ace['matches']['tcp-acl']['ietf-mud:direction-initiated']
                                        source = ace['matches']['ipv6-acl']['ietf-acldns:src-dnsname']
                                        protocol = ace['matches']['ipv6-acl']['protocol']
                                        sports_start=None
                                        sports_end=None
                                        dports_start=None
                                        dports_end=None
                                        action=ace['actions']['forwarding']
                                        if ace['matches']['ipv6-acl']['source-port-range'] is not None:
                                            sports_start= ace['matches']['ipv6-acl']['source-port-range']['lower-port']
                                            sports_end= ace['matches']['ipv6-acl']['source-port-range']['upper-port']

                                        sports = (0,65535)
                                        dports = (0,65535)
                                        if sports_start is not None and sports_end is not None:
                                            sports=(sports_start,sports_end)
                                        if dports_start is not None and dports_end is not None:
                                            dports=(dports_start,dports_end)

                                        ace = ACE(source,dest,protocol,dports,sports,action)
                                        if 'to' not in acl_details:
                                            acl_details['to']=[]

                                        acl_details['to'].append(ace)


            #print(extracted['ietf-mud:mud']['from-device-policy']['access-lists']['access-list'][0]['acl-name'])
            #print(extracted['ietf-access-control-list:access-lists']['acl'][0]['access-list-entries']['ace'][0]['matches']['ipv6-acl']['ietf-acldns:src-dnsname'])

            return acl_details

    @staticmethod
    def lookup_acl_details_by_direction(extracted_data, direction_tag, acl_details):

        if extracted_data is not None:
            from_device_acls = extracted_data['ietf-mud:mud'][direction_tag]['access-lists']['access-list']
            if len(from_device_acls)>0:
                for acl in extracted_data['ietf-mud:mud'][direction_tag]['access-lists']['access-list']:
                    acl_name = acl['acl-name']
                    # lookup ACL data
                    data = extracted_data['ietf-access-control-list:access-lists']['acl']
                    if len(data)>0:
                        for acl_data in data:
                            if acl_name == acl_data['acl-name']:
                                aces= acl_data['access-list-entries']['ace']
                                if len(aces)>0:
                                    for ace in aces:
                                        dest=ace['matches']['tcp-acl']['ietf-mud:direction-initiated']
                                        source = ace['matches']['ipv6-acl']['ietf-acldns:src-dnsname']
                                        protocol = ace['matches']['ipv6-acl']['protocol']
                                        sports_start=None
                                        sports_end=None
                                        dports_start=None
                                        dports_end=None
                                        action=ace['actions']['forwarding']
                                        if ace['matches']['ipv6-acl']['source-port-range'] is not None:
                                            sports_start= ace['matches']['ipv6-acl']['source-port-range']['lower-port']
                                            sports_end= ace['matches']['ipv6-acl']['source-port-range']['upper-port']

                                        sports = (0,65535)
                                        dports = (0,65535)
                                        if sports_start is not None and sports_end is not None:
                                            sports=(sports_start,sports_end)
                                        if dports_start is not None and dports_end is not None:
                                            dports=(dports_start,dports_end)

                                        ace = ACE(source,dest,protocol,dports,sports,action)
                                        if 'from' in direction_tag:
                                            if 'from' not in acl_details:
                                                acl_details['from']=[]
                                            acl_details['from'].append(ace)
                                        elif 'to' in direction_tag:
                                            if 'to' not in acl_details:
                                                acl_details['to']=[]
                                            acl_details['to'].append(ace)


            #print(extracted['ietf-mud:mud']['from-device-policy']['access-lists']['access-list'][0]['acl-name'])
            #print(extracted['ietf-access-control-list:access-lists']['acl'][0]['access-list-entries']['ace'][0]['matches']['ipv6-acl']['ietf-acldns:src-dnsname'])

    @staticmethod
    def get_port_descriptor(protocol, port_tuple, type):
        port_descriptor=None
        if port_tuple is not None:
            if protocol==6:
                if port_tuple[0]!=port_tuple[1]:
                    port_descriptor = 'TCP.%s=%s-%s'%(type,port_tuple[0],port_tuple[1])
                else:
                    port_descriptor = 'TCP.%s=%s'%(type,port_tuple[0])

            elif protocol==17:
                if port_tuple[0]!=port_tuple[1]:
                    port_descriptor = 'UDP.%s=%s-%s'%(type,port_tuple[0],port_tuple[1])
                else:
                    port_descriptor = 'UDP.%s=%s'%(type,port_tuple[0])

        return port_descriptor

    @staticmethod
    def get_ipaddresses_numeric(ipaddress_list):
        import ipaddr
        numeric_ipaddresses=[]
        for ipaddress in ipaddress_list:
            if ipaddress=='0.0.0.0/0':
                end = int(ipaddr.IPv4Network('255.255.255.255/32'))
                value='0-%s'%end
            else:
                ipv4address = ipaddr.IPv4Network(ipaddress)
                value= '%s-%s'%(int(ipv4address.network), int(ipv4address.broadcast))
            if value not in numeric_ipaddresses:
                numeric_ipaddresses.append(value)

        return numeric_ipaddresses


    def get_associated_edges(self, element, edge_list):
         return [edge for edge in edge_list if element in edge.invertex]

    def get_pathways_to(self, source, target, metagraph):
        # compute all metapaths
        metapaths = metagraph.get_all_metapaths_from(source,target,True)

        # extract dominant metapaths
        extracted=[]
        if metapaths is not None and len(metapaths)>0:
            for mp in metapaths:
                if metagraph.is_edge_dominant_metapath(mp) and \
                   self.is_metapath_absent(mp,extracted):
                   extracted.append(mp)

        return extracted

    def is_metapath_absent(self, mp, mp_list):
        for mp1 in mp_list:
            if self.is_edge_list_included(mp.edge_list, mp1.edge_list):
                return False

        return True

    def get_edge_attributes(self, key, attr_list):
        result=[]
        for attr in attr_list:
            if key in attr: result.append(attr)

        return set(result)



























