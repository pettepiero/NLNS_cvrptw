import numpy as np
import torch
import logging

def get_distances_matrix(locations):
    n = len(locations)
    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            m[i, j] = np.sqrt( (locations[i][0] - locations[j][0])**2 + (locations[i][1] - locations[j][1])**2) 

    return m

log = logging.getLogger(__name__)
class VRPInstance():
    def __init__(self, nb_customers, locations, original_locations, demand, capacity, time_window, service_time, max_time, late_coeff=1, use_cost_memory=True, schedule=None):
        self.nb_customers = nb_customers
        self.locations = locations  # coordinates of all locations in the interval [0, 1]
        self.original_locations = original_locations  # original coordinates of locations (used to compute objective
        # value)
        self.demand = demand  # demand for each customer (integer). Values are divided by capacity right before being
        # fed to the network
        self.capacity = capacity  # capacity of the vehicle
        self.time_window = time_window # time window of each customer, expressed as tuple of integers
        self.service_time = service_time
        #self.early_coeff = early_coeff
        self.late_coeff = late_coeff

        self.solution = None  # List of tours. Each tour is a list of location elements. Each location element is a
        # list with three values [i_l, d, i_n], with i_l being the index of the location (0=depot),
        # d being the fulfilled demand of customer i_l by that tour, and
        # i_n being the index of the associated network input.

        self.schedule = schedule
        #self.speed_f = 1000
        self.speed_f = 1

        self.nn_input_idx_to_tour = None  # After get_network_input() has been called this is a list where the
        # i-th element corresponds to the tour end represented by the i-th network input. If the network
        # points to an input, this allows us to find out, which tour end that input corresponds to.
        self.open_nn_input_idx = None  # List of idx of those nn_inputs that have not been visited
        self.incomplete_tours = None  # List of incomplete tours of self.solution
        if use_cost_memory:
            self.costs_memory = np.full((nb_customers + 1, nb_customers + 1), np.nan, dtype="float")
        else:
            self.costs_memory = None

        self.distances = get_distances_matrix(self.locations)
        self.depot_indices = [0]
        self.customer_indices = [i for i in range(1, self.nb_customers)]
        self.max_time = max_time

    def get_n_closest_locations_to(self, origin_location_id, mask, n):
        """Return the idx of the n closest locations sorted by distance."""
        distances = np.array([np.inf] * len(mask))
        distances[mask] = ((self.locations[mask] * self.locations[origin_location_id]) ** 2).sum(1)
        order = np.argsort(distances)
        return order[:n]

    def create_initial_solution(self):
        """Create an initial solution for this instance using a greedy heuristic."""
        #self.solution = [[[0, 0, 0]], [[0, 0, 0]]]
        self.solution = [[[0, 0, 0]]]
        cur_load = self.capacity
        visited_mask = np.array([True] * (self.nb_customers + 1))
        visited_mask[0] = False
        t = 0
        while visited_mask.any():
            # update mask for customers that are reachable in time
            current_tour = [[0, 0, 0]]
            self.solution.append(current_tour)
            cur_load = self.capacity
            t = 0
            current_node = 0

            while True:
                custs_indices = np.arange(self.nb_customers +1)
                travel_times = self.speed_f * self.distances[current_node][custs_indices]
                arrival_times = t + travel_times
                in_time = (arrival_times <= self.time_window[:, 1])
                temp_mask = visited_mask & in_time

                if not temp_mask.any():
                    break

                closest_customer_idx = self.get_n_closest_locations_to(current_node, temp_mask, 1)[0]

                if self.demand[closest_customer_idx] <= cur_load:
                    # add to tour
                    visited_mask[closest_customer_idx] = False
                    current_tour.append([int(closest_customer_idx), int(self.demand[closest_customer_idx]), None])

                    arrival = t + self.speed_f * self.distances[current_node][closest_customer_idx]
                    start_service = max(arrival, self.time_window[closest_customer_idx][0])
                    t = start_service + self.service_time
                    cur_load -= self.demand[closest_customer_idx]
                    current_node = closest_customer_idx
                else:
                    # vehicle is full even if customer was in time
                    break
            self.solution[-1].append([0, 0, 0])

        #create schedule
        self.schedule = []
        for i, tour in enumerate(self.solution):
            if len(tour) > 1:
                self.schedule.append(self.compute_tour_schedule(tour))
            else:
                self.schedule.append([[0,0]])

    def get_costs_memory(self, round):
        """Return the cost of the current complete solution. Uses a memory to improve performance."""
        c = 0
        late_mins   = 0
        for route_idx, t in enumerate(self.solution):
            if t[0][0] != 0 or t[-1][0] != 0:
                raise Exception("Incomplete solution.")
            for i in range(0, len(t) - 1):
                from_idx = t[i][0]
                to_idx = t[i + 1][0]
                if np.isnan(self.costs_memory[from_idx, to_idx]):
                    cc = np.sqrt((self.original_locations[from_idx, 0] - self.original_locations[to_idx, 0]) ** 2
                                 + (self.original_locations[from_idx, 1] - self.original_locations[to_idx, 1]) ** 2)
                    if round:
                        cc = np.round(cc)
                    self.costs_memory[from_idx, to_idx] = cc
                    c += cc
                else:
                    c += self.costs_memory[from_idx, to_idx]

                late_mins   += max(0, self.schedule[route_idx][i][1] - self.time_window[from_idx][1])
                c += self.late_coeff*late_mins
        return c

    def get_costs(self, round):
        """Return the cost of the current complete solution."""
        c = 0
        late_mins   = 0
        for route_idx, t in enumerate(self.solution):
            if t[0][0] != 0 or t[-1][0] != 0:
                raise Exception("Incomplete solution.")
            for i in range(0, len(t) - 1):
                from_idx = t[i][0]
                to_idx = t[i + 1][0]
                cc = np.sqrt((self.original_locations[t[i][0], 0] - self.original_locations[t[i + 1][0], 0]) ** 2
                             + (self.original_locations[t[i][0], 1] - self.original_locations[t[i + 1][0], 1]) ** 2)
                if round:
                    cc = np.round(cc)
                c += cc
                late_mins = max(0, self.schedule[route_idx][i][1] - self.time_window[from_idx][1])
                c += self.late_coeff*late_mins
        return c

    def get_costs_incomplete(self, round):
        """Return the cost of the current incomplete solution."""
        c = 0
        late_mins   = 0
        for route_idx, tour in enumerate(self.solution):
            if len(tour) <= 1:
                continue
            for i in range(0, len(tour) - 1):
                from_idx = tour[i][0]
                to_idx = tour[i + 1][0]
                cc = np.sqrt((self.original_locations[tour[i][0], 0] - self.original_locations[tour[i + 1][0], 0]) ** 2
                             + (self.original_locations[tour[i][0], 1] - self.original_locations[
                    tour[i + 1][0], 1]) ** 2)
                if round:
                    cc = np.round(cc)
                c += cc
                late_mins = max(0, self.schedule[route_idx][i][1] - self.time_window[from_idx][1])
                c += self.late_coeff*late_mins
        return c

    def get_total_distance(self, round_dist=False):
        """Return the distance of the current complete solution."""
        c = 0
        for route_idx, t in enumerate(self.solution):
            if t[0][0] != 0 or t[-1][0] != 0:
                raise Exception("Incomplete solution.")
            for i in range(0, len(t) - 1):
                from_idx = t[i][0]
                to_idx = t[i + 1][0]
                cc = np.sqrt((self.original_locations[t[i][0], 0] - self.original_locations[t[i + 1][0], 0]) ** 2
                             + (self.original_locations[t[i][0], 1] - self.original_locations[t[i + 1][0], 1]) ** 2)
                if round_dist:
                    cc = np.round(cc)
                c += cc
        return c

    def get_sum_late_mins(self):
        """Returns the sum of late mins according to computed schedule"""
        sum_late_mins = 0
        schedule = []
        for t in self.solution:
            if len(t) > 1:
                schedule.append(self.compute_tour_schedule(t))
            else:
                schedule.append([[0, 0]])

        for t, s in zip(self.solution, schedule):
            for i in range(len(t)-1):
                cust = t[i][0] 
                tw_end = self.time_window[cust][1]
                delay = max(0, s[i][1] - tw_end)
                sum_late_mins += delay

        return sum_late_mins 

    def destroy(self, customers_to_remove_idx):
        """Remove the customers with the given idx from their tours. This creates an incomplete solution."""
        self.incomplete_tours = []
        st = []  # solution tours
        sc = []  # solution schedule

        removed_customer_idx = []

        for tour_idx, tour in enumerate(self.solution):
            assert len(self.schedule) == len(self.solution)
            last_split_idx = 0
            for i in range(1, len(tour) - 1):
                customer_idx = tour[i][0]
                if customer_idx in customers_to_remove_idx:
                    # Create two new tours:
                    # The first consisting of the tour from the depot or from the last removed customer to the
                    # customer that should be removed
                    if i > last_split_idx and i > 1:
                        new_tour_pre = tour[last_split_idx:i]
                        st.append(new_tour_pre)
                        sc.append(self.schedule[tour_idx][last_split_idx:i])
                        self.incomplete_tours.append(new_tour_pre)

                    # The second consisting of only the customer to be removed
                    if customer_idx not in removed_customer_idx:  # make sure the customer has not already been
                        # extracted from a different tour
                        demand = int(self.demand[customer_idx])
                        new_tour = [[customer_idx, demand, None]]
                        st.append(new_tour)
                        time = int(max(0, self.time_window[customer_idx][0] - self.speed_f*self.distances[0][customer_idx]))
                        sc.append([[time, time + self.service_time]]) 
                        self.incomplete_tours.append(new_tour)
                        removed_customer_idx.append(customer_idx)
                    last_split_idx = i + 1

            if last_split_idx > 0:
                # Create another new tour consisting of the remaining part of the original tour
                if last_split_idx < len(tour) - 1:
                    new_tour_post = tour[last_split_idx:]
                    st.append(new_tour_post)
                    sc.append(self.compute_tour_schedule(new_tour_post))
                    self.incomplete_tours.append(new_tour_post)
            else:  # add unchanged tour
                st.append(tour)
                sc.append(self.schedule[tour_idx])

        self.solution = st
        self.schedule = sc

    def compute_tour_schedule(self, tour):
        schedule = []
        depart_time_current = 0
        current_cust = None 
        for i in range(len(tour) -1):
            current_cust = tour[i][0]
            next_cust = tour[i+1][0]
            travel_time = self.speed_f * self.distances[current_cust][next_cust]
            arrival = depart_time_current + travel_time

            tw_open, tw_close = self.time_window[next_cust]
            start_service = max(arrival, tw_open)

            depart = start_service + self.service_time
            schedule.append([int(arrival), int(depart)])
            depart_time_current = depart

        last_cust = tour[-1][0]
        travel_time = self.speed_f * self.distances[current_cust][last_cust]
        schedule.append([int(depart_time_current + travel_time), self.max_time]) # set max_time to 1000
        assert len(tour) == len(schedule)
        return schedule

    def destroy_random(self, p, rng):
        """Random destroy. Select customers that should be removed at random and remove them from tours."""
        customers_to_remove_idx = rng.choice(range(1, self.nb_customers + 1), int(self.nb_customers * p),
                                                   replace=False)
        self.destroy(customers_to_remove_idx)

    def destroy_point_based(self, p, rng):
        """Point based destroy. Select customers that should be removed based on their distance to a random point
         and remove them from tours."""
        nb_customers_to_remove = int(self.nb_customers * p)
        random_point = rng.random((1, 2))
        dist = np.sum((self.locations[1:] - random_point) ** 2, axis=1)
        closest_customers_idx = np.argsort(dist)[:nb_customers_to_remove] + 1
        self.destroy(closest_customers_idx)

    def destroy_tour_based(self, p, rng):
        """Tour based destroy. Remove all tours closest to a randomly selected point from a solution."""
        # Make a dictionary that maps customers to tours
        customer_to_tour = {}
        for i, tour in enumerate(self.solution[1:]):
            for e in tour[1:-1]:
                if e[0] in customer_to_tour:
                    customer_to_tour[e[0]].append(i + 1)
                else:
                    customer_to_tour[e[0]] = [i + 1]

        nb_customers_to_remove = int(self.nb_customers * p)  # Number of customer that should be removed
        nb_removed_customers = 0
        tours_to_remove_idx = []
        random_point = rng.random((1, 2))  # Randomly selected point
        dist = np.sum((self.locations[1:] - random_point) ** 2, axis=1)
        closest_customers_idx = np.argsort(dist) + 1

        # Iterate over customers starting with the customer closest to the random point.
        for customer_idx in closest_customers_idx:
            # Iterate over the tours of the customer
            for i in customer_to_tour[customer_idx]:
                # and if the tour is not yet marked for removal
                if i not in tours_to_remove_idx:
                    # mark it for removal
                    tours_to_remove_idx.append(i)
                    nb_removed_customers += len(self.solution[i])

            # Stop once enough tours are marked for removal
            if nb_removed_customers >= nb_customers_to_remove and len(tours_to_remove_idx) > 1:
                break

        # Create the new tours that all consist of only a single customer
        new_tours = []
        new_schedules = []
        removed_customer_idx = []
        for i in tours_to_remove_idx:
            tour = self.solution[i]
            for e in tour[1:-1]:
                #if e[0] in removed_customer_idx:
                #    for new_tour in new_tours:
                #        if new_tour[0][0] == e[0]:
                #            new_tour[0][1] += e[1]
                #            break
                #else:
                #    new_tours.append([e])
                #    time = int(max(0, self.time_window[e][0] - self.speed_f*self.distances[0][e]))
                #    new_schedules.append([[time, time + self.service_time]]) 
                #    removed_customer_idx.append(e[0])
                new_tours.append([e])
                time = int(max(0, self.time_window[e[0]][0] - self.speed_f*self.distances[0][e[0]]))
                new_schedules.append([[time, time + self.service_time]]) 
                removed_customer_idx.append(e[0])

        # Remove the tours that are marked for removal from the solution
        for index in sorted(tours_to_remove_idx, reverse=True):
            del self.solution[index]
            del self.schedule[index]

        self.solution.extend(new_tours)  # Add new tours to solution
        self.schedule.extend(new_schedules)
        self.incomplete_tours = new_tours

    def _get_incomplete_tours(self):
        incomplete_tours = []
        for tour in self.solution:
            if tour[0][0] != 0 or tour[-1][0] != 0:
                incomplete_tours.append(tour)
        return incomplete_tours

    def get_max_nb_input_points(self):
        incomplete_tours = self.incomplete_tours
        nb = 1  # input point for the depot
        for tour in incomplete_tours:
            if len(tour) == 1:
                nb += 1
            else:
                if tour[0][0] != 0:
                    nb += 1
                if tour[-1][0] != 0:
                    nb += 1
        return nb

    def get_network_input(self, input_size):
        """Generate the tensor representation of an incomplete solution (i.e, a representation of the repair problem).
         The input size must be provided so that the representations of all inputs of the batch have the same size.

        [:, 0] x-coordinates for all points
        [:, 1] y-coordinates for all points
        [:, 2] start of time window for all points
        [:, 3] end of time window for all points
        #added later:
        #[:, 4] demand values for all points
        #[:, 5] state values for all points
        
        # old notation
        #[:, 0] x-coordinates for all points
        #[:, 1] y-coordinates for all points
        #[:, 2] demand values for all points
        #[:, 3] state values for all points
        """
        nn_input = np.zeros((input_size, 6))
        nn_input[0, :2] = self.locations[0]  # Depot location
        nn_input[0, 2] = 0
        nn_input[0, 3] = self.max_time / self.max_time
        nn_input[0, 4] = -1 * self.capacity  # Depot demand
        nn_input[0, 5] = -1  # Depot state
        network_input_idx_to_tour = [None] * input_size
        network_input_idx_to_tour[0] = [self.solution[0], 0]
        i = 1
        destroyed_location_idx = []

        incomplete_tours = self.incomplete_tours
        for tour in incomplete_tours:
            # Create an input for a tour consisting of a single customer
            if len(tour) == 1:
                nn_input[i, :2] = self.locations[tour[0][0]]
                nn_input[i, 2] = self.time_window[tour[0][0]][0] / self.max_time
                nn_input[i, 3] = self.time_window[tour[0][0]][1] / self.max_time
                nn_input[i, 4] = tour[0][1]
                nn_input[i, 5] = 1
                tour[0][2] = i
                network_input_idx_to_tour[i] = [tour, 0]
                destroyed_location_idx.append(tour[0][0])
                i += 1
            else:
                # Create an input for the first location in an incomplete tour if the location is not the depot
                if tour[0][0] != 0:
                    nn_input[i, :2] = self.locations[tour[0][0]]
                    nn_input[i, 2] = self.time_window[tour[0][0]][0] / self.max_time
                    nn_input[i, 3] = self.time_window[tour[0][0]][1] / self.max_time
                    nn_input[i, 4] = sum(l[1] for l in tour)
                    network_input_idx_to_tour[i] = [tour, 0]
                    if tour[-1][0] == 0:
                        nn_input[i, 5] = 3
                    else:
                        nn_input[i, 5] = 2
                    tour[0][2] = i
                    destroyed_location_idx.append(tour[0][0])
                    i += 1
                # Create an input for the last location in an incomplete tour if the location is not the depot
                if tour[-1][0] != 0:
                    nn_input[i, :2] = self.locations[tour[-1][0]]
                    nn_input[i, 2] = self.time_window[tour[-1][0]][0] / self.max_time
                    nn_input[i, 3] = self.time_window[tour[-1][0]][1] / self.max_time
                    nn_input[i, 4] = sum(l[1] for l in tour)
                    network_input_idx_to_tour[i] = [tour, len(tour) - 1]
                    tour[-1][2] = i
                    if tour[0][0] == 0:
                        nn_input[i, 5] = 3
                    else:
                        nn_input[i, 5] = 2
                    destroyed_location_idx.append(tour[-1][0])
                    i += 1

        self.open_nn_input_idx = list(range(1, i))
        self.nn_input_idx_to_tour = network_input_idx_to_tour
        return nn_input[:, :4], nn_input[:, 4:]

    def _get_network_input_update_for_tour(self, tour, new_demand):
        """Returns an nn_input update for the tour tour. The demand of the tour is updated to new_demand"""
        nn_input_idx_start = tour[0][2]  # Idx of the nn_input for the first location in tour
        nn_input_idx_end = tour[-1][2]  # Idx of the nn_input for the last location in tour

        # If the tour stars and ends at the depot, no update is required
        if nn_input_idx_start == 0 and nn_input_idx_end == 0:
            return []

        nn_input_update = []
        # Tour with a single location
        if len(tour) == 1:
            if tour[0][0] != 0:
                nn_input_update.append([nn_input_idx_end, new_demand, 1])
                self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, 0]
        else:
            # Tour contains the depot
            if tour[0][0] == 0 or tour[-1][0] == 0:
                # First location in the tour is not the depot
                if tour[0][0] != 0:
                    nn_input_update.append([nn_input_idx_start, new_demand, 3])
                    # update first location
                    self.nn_input_idx_to_tour[nn_input_idx_start] = [tour, 0]
                # Last location in the tour is not the depot
                elif tour[-1][0] != 0:
                    nn_input_update.append([nn_input_idx_end, new_demand, 3])
                    # update last location
                    self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, len(tour) - 1]
            # Tour does not contain the depot
            else:
                # update first and last location of the tour
                nn_input_update.append([nn_input_idx_start, new_demand, 2])
                self.nn_input_idx_to_tour[nn_input_idx_start] = [tour, 0]
                nn_input_update.append([nn_input_idx_end, new_demand, 2])
                self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, len(tour) - 1]
        return nn_input_update

    def do_action(self, id_from, id_to):
        """Performs an action. The tour end represented by input with the id id_from is connected to the tour end
         presented by the input with id id_to."""
        #log.info(f"\n\nin do_action | id_from: {id_from}, id_to {id_to}")
        #log.info(f"in do_action | self.solution:")
        #for el in self.solution:
        #    log.info(el)
        tour_from = self.nn_input_idx_to_tour[id_from][0]  # Tour that should be connected
        tour_to = self.nn_input_idx_to_tour[id_to][0]  # to this tour.
        #log.info(f"in do_action | tour_from: {tour_from}, tour_to {tour_to}")
        pos_from = self.nn_input_idx_to_tour[id_from][1]  # Position of the location that should be connected in tour_from
        pos_to = self.nn_input_idx_to_tour[id_to][1]  # Position of the location that should be connected in tour_to
        #log.info(f"in do_action | pos_from: {pos_from}, pos_to {pos_to}")

        nn_input_update = []  # Instead of recalculating the tensor representation, we only compute an update description.
        # This improves performance.

        # Exchange tour_from with tour_to or invert order of the tours. This reduces the number of cases that need
        # to be considered in the following.
        if len(tour_from) > 1 and len(tour_to) > 1:
            if pos_from > 0 and pos_to > 0:
                tour_to.reverse()
            elif pos_from == 0 and pos_to == 0:
                tour_from.reverse()
            elif pos_from == 0 and pos_to > 0:
                tour_from, tour_to = tour_to, tour_from
        elif len(tour_to) > 1:
            if pos_to == 0:
                tour_to.reverse()
            tour_from, tour_to = tour_to, tour_from
        elif len(tour_from) > 1 and pos_from == 0:
            tour_from.reverse()
        #log.info(f"in do_action, after changing orders | tour_from: {tour_from}, tour_to {tour_to}")
        # Now we only need to consider two cases 1) Connecting an incomplete tour with more than one location
        # to an incomplete tour with more than one location 2) Connecting an incomplete tour (single
        # or multiple locations) to incomplete tour consisting of a single location

        # Case 1
        if len(tour_from) > 1 and len(tour_to) > 1:
            #log.info(f"in do_action | case #1")
            combined_demand = sum(l[1] for l in tour_from) + sum(l[1] for l in tour_to)
            #log.info(f"in do_action | combined_demand = {combined_demand} >? self.capacity: {combined_demand > self.capacity}")
            if combined_demand > self.capacity:
                log.info(f"\n\n**************************************************\n"
                f"Failed: logged to {get_logger_file_name()}\n\n")
            assert combined_demand <= self.capacity  # This is ensured by the masking schema

            # The two incomplete tours are combined to one (in)complete tour. All network inputs associated with the
            # two connected tour ends are set to 0
            nn_input_update.append([tour_from[-1][2], 0, 0])
            nn_input_update.append([tour_to[0][2], 0, 0])
            tour_from.extend(tour_to)
            self.solution.remove(tour_to)
            nn_input_update.extend(self._get_network_input_update_for_tour(tour_from, combined_demand))

        # Case 2
        if len(tour_to) == 1:
            #log.info(f"in do_action | case #2")
            demand_from = sum(l[1] for l in tour_from)
            combined_demand = demand_from + sum(l[1] for l in tour_to)
            unfulfilled_demand = combined_demand - self.capacity

            # The new tour has a total demand that is smaller than or equal to the vehicle capacity
            if unfulfilled_demand <= 0:
                if len(tour_from) > 1:
                    nn_input_update.append([tour_from[-1][2], 0, 0])
                # Update solution
                tour_from.extend(tour_to)
                self.solution.remove(tour_to)
                # Generate input update
                nn_input_update.extend(self._get_network_input_update_for_tour(tour_from, combined_demand))
            # The new tour has a total demand that is larger than the vehicle capacity
            else:
                nn_input_update.append([tour_from[-1][2], 0, 0])
                if len(tour_from) > 1 and tour_from[0][0] != 0:
                    nn_input_update.append([tour_from[0][2], 0, 0])

                # Update solution
                tour_from.append([tour_to[0][0], tour_to[0][1], tour_to[0][2]])  # deepcopy of tour_to
                tour_from[-1][1] = self.capacity - demand_from
                tour_from.append([0, 0, 0])
                if tour_from[0][0] != 0:
                    tour_from.insert(0, [0, 0, 0])
                tour_to[0][1] = unfulfilled_demand  # Update demand of tour_to

                nn_input_update.extend(self._get_network_input_update_for_tour(tour_to, unfulfilled_demand))

        # Add depot tour to the solution tours if it was removed
        if self.solution[0] != [[0, 0, 0]]:
            self.solution.insert(0, [[0, 0, 0]])
            self.nn_input_idx_to_tour[0] = [self.solution[0], 0]

        for update in nn_input_update:
            if update[2] == 0 and update[0] != 0:
                self.open_nn_input_idx.remove(update[0])

        #update schedules
        sc = []
        for tour in self.solution:
            if len(tour) > 1:
                sc.append(self.compute_tour_schedule(tour))
            else:
                sc.append([[0, 0]])
        self.schedule = sc

        return nn_input_update, tour_from[-1][2]

    def verify_solution(self, config):
        """Verify that a feasible solution has been found."""
        d = np.zeros((self.nb_customers + 1), dtype=int)
        for i in range(len(self.solution)):
            for ii in range(len(self.solution[i])):
                d[self.solution[i][ii][0]] += self.solution[i][ii][1]
        if (self.demand != d).any():
            raise Exception('Solution could not be verified.')

        for tour in self.solution:
            if sum([t[1] for t in tour]) > self.capacity:
                raise Exception('Solution could not be verified.')

        if not config.split_delivery:
            customers = []
            for tour in self.solution:
                for c in tour:
                    if c[0] != 0:
                        customers.append(c[0])

            if len(customers) > len(set(customers)):
                raise Exception('Solution could not be verified.')

    def get_solution_copy(self):
        """ Returns a copy of self.solution"""
        solution_copy = []
        for tour in self.solution:
            solution_copy.append([x[:] for x in tour]) # Fastest way to make a deep copy
        return solution_copy

    def __deepcopy__(self, memo):
        solution_copy = self.get_solution_copy()
        schedule_copy = [[x[:] for x in tour_sched] for tour_sched in self.schedule]
        new_instance = VRPInstance(self.nb_customers, self.locations, self.original_locations, self.demand,
                                   self.capacity, self.time_window, self.service_time, schedule=schedule_copy, max_time=self.max_time, late_coeff=self.late_coeff)
        new_instance.solution = solution_copy
        new_instance.costs_memory = self.costs_memory

        return new_instance

    def get_last_dim(self, static_input, origin_idx):
        # (N, 2) coords in NN-input space
        coords = static_input[:, :2]
        origin_xy = coords[origin_idx]  # (2,)
        distances = torch.sqrt(((coords - origin_xy) ** 2).sum(dim=-1))  # (N,)
        origin_tour, origin_pos_in_tour = self.nn_input_idx_to_tour[origin_idx]
        tour_idx = self.solution.index(origin_tour)
        current_time = self.schedule[tour_idx][origin_pos_in_tour][1]  # should be scalar (int/float)
        #scale down current_time to 0,1
        current_time = current_time / self.max_time 
        last_dim = distances.clone()
        last_dim[origin_idx] = float(current_time)

        return last_dim.unsqueeze(-1)  # (N, 1)
    

    def _check_alignment(self, where=""):
        if self.solution is None or self.schedule is None:
            return
        if len(self.solution) != len(self.schedule):
            print("ALIGNMENT BROKEN", where)
            print("len(solution)=", len(self.solution), "len(schedule)=", len(self.schedule))
            raise RuntimeError("schedule/solution misaligned")
    
    #def compute_time_feasibility(self, origin_nn_idx, target_nn_idx):
    #    origin_tour, origin_pos_in_tour = self.nn_input_idx_to_tour[origin_nn_idx]
    #    target_tour, target_pos_in_tour = self.nn_input_idx_to_tour[target_nn_idx]
    #    
    #    if origin_tour is None or target_tour is None:
    #        return False

    #    origin_node_id = origin_tour[origin_pos_in_tour][0]
    #    target_node_id = target_tour[target_pos_in_tour][0]

    #    current_time = self.schedule[self.solution.index(origin_tour)][origin_pos_in_tour][1]
    #    travel_time = self.speed_f*self.distances[origin_node_id, target_node_id]
    #    arrival_at_target = current_time + travel_time

    #    latest_arrival = self.time_window[target_node_id][1]

    #    return arrival_at_target <= latest_arrival


def get_mask(origin_nn_input_idx, static_input, dynamic_input, instances, config, capacity):
    """ Returns a mask for the current nn_input"""
    batch_size = origin_nn_input_idx.shape[0]
    #log.info(f"\n\tin get_mask | origin_nn_input_idx = {origin_nn_input_idx}")
    #log.info(f"\tin get_mask | batch_size = {batch_size}")

    last_dims = dynamic_input[:, :, -1]
    #log.info(f"\tin get_mask | last_dims = {last_dims}")
    current_times = last_dims[torch.arange(last_dims.size(0)), origin_nn_input_idx]
    #log.info(f"\tin get_mask | current_times = {current_times}")
    speed_f = [ins.speed_f for ins in instances]
    speed_f = torch.tensor(speed_f, dtype=float, device=last_dims.device)
    arrival_times = last_dims[torch.arange(last_dims.size(0)), :]*speed_f.unsqueeze(-1) + current_times.unsqueeze(-1)
    #log.info(f"\tin get_mask | arrival_times = {arrival_times}")
    tw_end = static_input[:,:,3]
    #log.info(f"\tin get_mask | tw_end = {tw_end}")

    time_feasible = (arrival_times <= tw_end).cpu().numpy().astype(int)
    #log.info(f"\tin get_mask | time_feasible = {time_feasible}")

    # Start with all used input positions
    mask = (dynamic_input[:, :, 1] != 0).cpu().long().numpy()
    mask = mask * time_feasible
    #log.info(f"\tin get_mask | mask = {mask}")

    for i in range(batch_size):
        idx_from = origin_nn_input_idx[i]
        origin_tour = instances[i].nn_input_idx_to_tour[idx_from][0]
        origin_pos = instances[i].nn_input_idx_to_tour[idx_from][1]

        # Find the start of the tour in the nn input
        # e.g. for the tour [2, 3] two entries in nn input exists
        if origin_pos == 0:
            idx_same_tour = origin_tour[-1][2]
        else:
            idx_same_tour = origin_tour[0][2]

        mask[i, idx_same_tour] = 0

        # Do not allow origin location = destination location
        mask[i, idx_from] = 0

    mask = torch.from_numpy(mask)
    #log.info(f"\tin get_mask | mask = {mask}")

    origin_tour_demands = dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 0]
    #log.info(f"\tin get_mask | origin_tour_demands: {origin_tour_demands}")
    combined_demand = origin_tour_demands.unsqueeze(1).expand(batch_size, dynamic_input.shape[1]) + dynamic_input[:, :,0]
    #log.info(f"\tin get_mask | combined_demand: {combined_demand}")

    if config.split_delivery:
        raise NotImplementedError #not implemented for cvrptw
        multiple_customer_tour = (dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 1] > 1).unsqueeze(1).expand(
            batch_size, dynamic_input.shape[1])

        # If the origin tour consists of multiple customers mask all tours with multiple customers where
        # the combined demand is > 1
        mask[multiple_customer_tour & (combined_demand > capacity) & (dynamic_input[:, :, 1] > 1)] = 0

        # If the origin tour consists of a single customer mask all tours with demand is >= 1
        mask[(~multiple_customer_tour) & (dynamic_input[:, :, 0] >= capacity)] = 0
    else:
        mask[combined_demand > capacity] = 0

    mask[:, 0] = 1  # Always allow to go to the depot
    #log.info(f"\tin get_mask | mask = {mask}")

    return mask



def get_logger_file_name():
    # 1. Look for a FileHandler in the current logger or its parents (root)
    log_file = "Console/Unknown"
    current_logger = log
    while current_logger:
        for handler in current_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = handler.baseFilename
                break
        if log_file != "Console/Unknown" or not current_logger.propagate:
            break
        current_logger = current_logger.parent
    return str(log_file)
