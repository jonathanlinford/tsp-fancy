#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        bssf = None
        count = 0

        start_time = time.time()

        route = [cities[0]]
        currentCityIndex = 0

        # while all of the cities aren't added to the route
        while (len(route) < ncities) and (time.time() - start_time < time_allowance):
            minIndex = 0
            minValue = math.inf

            # loop over all of the cities to find the best city to go to from there
            for i in range(0, ncities):
                # make sure it's not ourselves
                if currentCityIndex != i:
                    # make sure we haven't already been there
                    if not route.__contains__(cities[i]):
                        tempValue = cities[currentCityIndex].costTo(cities[i])

                        if tempValue < minValue:
                            minValue = tempValue
                            minIndex = i

            # add the city with the cheapest route from the current city
            currentCityIndex = minIndex
            route.append(cities[currentCityIndex])

        bssf = TSPSolution(route)
        end_time = time.time()

        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    def branchAndBound(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = self.defaultRandomTour(time_allowance=60.0)['soln']
        pruned = 0
        max_q_size = 0
        num_states = 0
        Q = []
        start_time = time.time()

        matrix = np.empty(shape=(len(cities), len(cities)), dtype=float)

        root = TSPNode(0, matrix, [], 0)

        # Fill with distances
        for i, c1 in enumerate(cities):
            for j, c2 in enumerate(cities):
                root.m[i, j] = c1.costTo(c2)
        # print(root.m)
        root.reduceMatrix(0, 0)
        root.addCityAndUpdateCost(cities[0])
        # print("ROOT")
        # print(root.m)

        # Add to heap
        heapq.heappush(Q, root)

        while len(Q) != 0 and time.time() - start_time < time_allowance:
            node = heapq.heappop(Q)

            for i, dist in enumerate(node.m[node.city._index]):
                num_states += 1

                # Hasn't been visited
                if dist != np.inf and cities[i] not in node.route:
                    # Create and fill in node
                    next_node = TSPNode(node.lower_bound, np.copy(node.m), node.route.copy(), node.cost)
                    next_node.addCityAndUpdateCost(cities[i])
                    next_node.reduceMatrix(node.city._index, next_node.city._index)

                    # if cities all accounted for
                    if len(next_node.route) == len(cities):
                        solution = TSPSolution(next_node.route)
                        # see if cost is better
                        if solution.cost < bssf.cost:
                            count += 1
                            bssf = solution
                    else:  # if a solution hasn't been found yet

                        # should I prune
                        if next_node.lower_bound < bssf.cost:
                            heapq.heappush(Q, next_node)
                            max_q_size = max(max_q_size, len(Q))
                        else:
                            pruned += 1

        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = max_q_size
        results['total'] = num_states
        results['pruned'] = pruned
        return results

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = self.defaultRandomTour(time_allowance=10.0)['soln']
        pruned = 0
        max_q_size = 0
        num_states = 0
        start_time = time.time()

        tabu_list = []
        tabu_max_size = 50 # I chose an arbitrary size. We may want to tinker with this, and it may vary from scenario size to scenario size

        while (time.time() - start_time < time_allowance):
            candidates = []
            for candidate in get_neighborhood(bssf):
                if features_match(candidate, tabu_list):
                    candidates.append(candidate)
            current_candidate = get_best_candidate(candidates)
            if current_candidate.cost() < bssf.cost:
                prev_bssf = bssf
                bssf = current_candidate
                tabu_list = feature_difference(bssf, prev_bssf)
                while len(tabu_list) > tabu_max_size:
                    tabu_list.pop()

        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = max_q_size
        results['total'] = num_states
        results['pruned'] = pruned
        return results


def get_neighborhood(solution):
    pass


def features_match(solution_candidate, tabu_list):
    return false


def get_best_candidate(candidates):
    min_cost = np.inf
    output_candidate = None
    for candidate in candidates:
        if candidate.cost < min_cost:
            min_cost = candidate.cost
            output_candidate candidate
    return output_candidate


def feature_difference(current_best_solution, previous_best_solution):
    # should return a new feature list
    pass
