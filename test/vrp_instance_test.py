import unittest
from vrp.vrp_problem import VRPInstance, get_mask, get_backward_mask, get_forward_mask
from vrp.data_utils import read_instance_vrp, vrp_to_plot_solution
from pyvrp import read as pyvrp_read
from pyvrp.plotting import plot_solution
import numpy as np
import torch
from unittest.mock import Mock

class test_get_costs(unittest.TestCase):
    def setUp(self):
        self.instance = read_instance_vrp('./test/test_instance.vrp')

    def test_get_cost_complete(self):
        inst = self.instance
        self.assertEqual(inst.late_coeff, 1) 
        inst.create_initial_solution() 
        print("Instance initial solution:")
        for el in inst.solution:
            print(el)
        #print(f"Instance distances:")
        #for i, el in enumerate(inst.distances):
        #    print(f"From {i}")
        #    for j, el2 in enumerate(el):
        #        print(f"To {j}: {el2}")

        print("Instance schedule: ")
        for el in inst.schedule:
            print(el)
        print(f"Instance time window: ")
        for i, el in enumerate(inst.time_window):
            print(i, el)

        # Computing total delay of solution
        delay = 0
        for route_idx, route in enumerate(inst.solution):
            sched = inst.schedule[route_idx]
            for el_idx, el in enumerate(route):
                cust = el[0]
                arrival     = sched[el_idx][0]
                departure   = sched[el_idx][1]
                tw_end = inst.time_window[cust][1]
                delay += max(0, departure - tw_end) 

        self.assertEqual(delay, 0) # compare with hand computed delay

        # Computing total distance of solution
        dist = 0
        for route_idx, route in enumerate(inst.solution):
            for el_idx, el in enumerate(route[:-1]):
                cust = el[0]
                next_cust = route[el_idx+1][0]
                distance = np.sqrt((inst.original_locations[cust, 0] - inst.original_locations[next_cust, 0]) ** 2
                             + (inst.original_locations[cust, 1] - inst.original_locations[next_cust, 1]) ** 2)
                dist += distance

        self.assertTrue(abs(dist-inst.get_total_distance()) <= 0.1)
        total_cost = dist + inst.late_coeff*delay
        print(f"total_cost: {total_cost} = {dist} + {inst.late_coeff}*{delay} | inst.get_costs(False): {inst.get_costs(False)}")
        print(f"inst.get_costs(True): {inst.get_costs(True)}")
        self.assertTrue(abs(total_cost - inst.get_costs(False)) <= 0.001)
        self.assertTrue(abs(total_cost - inst.get_costs(True)) <= 5)
    
    def test_get_sum_late_mins(self):
        inst = self.instance
        inst.create_initial_solution() 
        self.assertEqual(inst.get_sum_late_mins(), 0)

class test_initial_solution(unittest.TestCase):
    def setUp(self):
        self.instance = read_instance_vrp('./test/test_instance.vrp')

    def test_create_initial_solution(self):
        inst = self.instance
        inst.create_initial_solution() 

        # Test lenghts of solution and schedule
        self.assertEqual(len(inst.solution), len(inst.schedule))
        for so, sc in zip(inst.solution, inst.schedule):
            self.assertEqual(len(so), len(sc))

class test_get_last_dim(unittest.TestCase):
    def setUp(self):
        self.instance = read_instance_vrp('./test/test_instance.vrp')

    def test_get_last_dim_values(self):
        inst = self.instance
        inst.create_initial_solution()

        # Make the instance "incomplete" so get_network_input() builds nn_input_idx_to_tour
        print(f"Initial solution:")
        for el in inst.solution:
            print(el)

        print(f"\n removing customer 5, which has coordinates: {inst.locations[5]}")
        # and open_nn_input_idx, which get_last_dim relies on.
        inst.destroy([5])  # remove customer 5 (pick any valid customer index)
        print(f"After destruction:")
        for el in inst.solution:
            print(el)
        print(f"Incomplete tours:")
        for el in inst.incomplete_tours:
            print(el)
        # Build the NN static input exactly the way repair does it
        input_size = inst.get_max_nb_input_points()
        static_np, _dynamic_np = inst.get_network_input(input_size)
        print(f"Equivalent nn_input_idx_to_tour:")
        for el in inst.nn_input_idx_to_tour:
            print(el)

        print(f"inst.locations | time_window:")
        for j, el in enumerate(zip(inst.locations, inst.time_window)):
            print(j, el)

        static_input = torch.tensor(static_np, dtype=torch.float32)
        print(f"Got static_input:")
        print(static_input)

        print(f"inst.demand:")
        for j, el in enumerate(inst.demand):
            print(j, el)
        print(f"Got _dynamic_np:")
        print(_dynamic_np)
        # Choose a valid origin idx (non-depot) that definitely exists
        origin_idx = inst.open_nn_input_idx[1]
        print(f"inst.open_nn_input_idx: {inst.open_nn_input_idx}")
        print(f"Using origin_idx: {origin_idx}")

        # Call the function under test
        out = inst.get_last_dim(static_input, origin_idx)

        # --- Assertions ---

        # 1) shape: (N, 1)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], static_input.shape[0])
        self.assertEqual(out.shape[1], 2)

        out_1d = out.squeeze(-1)

        # 2) expected_dists distances from origin to all points in coords space
        coords = static_input[:, :2]
        origin_xy = coords[origin_idx]
        expected_dists = torch.sqrt(((coords - origin_xy) ** 2).sum(dim=-1))

        # 3) expected_times override at origin_idx with scaled current_time
        tours = [el[0] for el in inst.nn_input_idx_to_tour]
        pos_in_tours = [el[1] for el in inst.nn_input_idx_to_tour]
        indices = list(map(inst.solution.index, tours))
        schedules = [inst.schedule[ind] for ind in indices]
        expected_times = []
        for j in range(len(indices)):
            if pos_in_tours[j] == 0:
                expected_times.append(schedules[j][pos_in_tours[j]][0])
            else:
                expected_times[j].append(schedules[j][pos_in_tours[j]][1])
        expected_times = torch.tensor(expected_times, dtype=expected_dists.dtype, device=expected_dists.device)
        print(f"DEBUG: expected_times: {expected_times}")
        print(f"DEBUG: expected_dists: {expected_dists}")

        expected = torch.stack((expected_dists, expected_times), dim=1)
        print(f"DEBUG: expected: {expected}")
        print(f"out_1d: {out_1d}")

        # 4) compare
        self.assertTrue(torch.allclose(out_1d, expected, atol=1e-6))


class test_get_mask(unittest.TestCase):
    def setUp(self):
        print(f"\nIn setUp()...")
        self.instance = read_instance_vrp('./test/test_instance.vrp')
        self.config = Mock()
        self.config.split_delivery = False
        self.config.device = 'cpu'
        inst = self.instance
        print(f"\n\tCreating initial solution...")
        inst.create_initial_solution()
        print(f"\t...done")
        print(f"\n\t Removing customer 5")
        inst.destroy([5])  # remove customer 5 (pick any valid customer index)
        print(f"\n\tAfter destruction:")
        for el in inst.solution:
            print('\t', el)
        print(f"\n\tAfter schedule:")
        for el in inst.schedule:
            print('\t', el)
        
        print(f"\n... finished setUp()")

    def test_get_backward_mask(self):
        print(f"\nTesting test_get_backward_mask()...")
        inst = self.instance

        origin_idx = 2
        print(f"\n\tChosing origin_idx: {origin_idx}")
        input_size = inst.get_max_nb_input_points()
        static_np, _dynamic_np = inst.get_network_input(input_size)
        print(f"\n\tDEBUG: inst.nn_input_idx_to_tour:")
        for j, el in enumerate(inst.nn_input_idx_to_tour):
            if j != origin_idx:
                print('\t  ', j, el)
            else:
                print('\t->', j, el)
        print(f"\n\torigin_idx={origin_idx} corresponds to open tour:")
        print('\t', inst.nn_input_idx_to_tour[origin_idx][0])
        print(f"\n\tSchedule with incomplete tour of nn_input at index {origin_idx}:")
        sched = inst.schedule[inst.solution.index(inst.nn_input_idx_to_tour[origin_idx][0])]
        pos = inst.nn_input_idx_to_tour[origin_idx][1]
        print('\t', sched) 
        if pos == 0:
            time = sched[pos][0] 
        else:
            time = sched[pos][1]
        print(f"\n\tThe tour end has time:", time) 
        custs_to_test = []
        current_tour, current_pos = inst.nn_input_idx_to_tour[origin_idx]
        current_cust = current_tour[current_pos][0]
        for t in inst.nn_input_idx_to_tour:
            tour, pos = t
            cust = tour[pos][0]
            custs_to_test.append(cust)
        print(f"\n\tCandidate customer indices: {custs_to_test}")
        print(f"\n\ttime_windows:")
        for j, el in enumerate(inst.time_window):
            if j not in custs_to_test:
                print('\t  ', j, el)
            else:
                print('\t->', j, el)
        handmade_dist_mask = torch.ones(len(custs_to_test), dtype=bool, device=self.config.device)
        for j, c in enumerate(custs_to_test):
            tw_open, tw_close = inst.time_window[c]
            travel_time = round(inst.distances[current_cust][c]*inst.speed_f, 2)
            print(f"\tDEBUG: travel_time from {current_cust} to {c}: {travel_time}")
            if current_pos == 0: # then candidates should go before current route
                # check backwards
                backward_arrival = round(time - travel_time - inst.service_time, 2)
                if tw_open <= backward_arrival:
                    handmade_dist_mask[j] = True
                    print(f"\tDEBUG: backward_arrival = {backward_arrival} vs tw_open = {tw_open}: \tFeasible")
                else:
                    handmade_dist_mask[j] = False
                    print(f"\tDEBUG: backward_arrival = {backward_arrival} vs tw_open = {tw_open}: \tUnfeasible")
            else:
                # check afterwards
                forward_arrival = round(time + travel_time + inst.service_time, 2)
                print(f"\tDEBUG: forward_arrival = {forward_arrival} vs tw_close = {tw_close}")
                if forward_arrival <= tw_close:
                    handmade_dist_mask[j] = True
                else:
                    handmade_dist_mask[j] = False

            if j == origin_idx:
                handmade_dist_mask[j] = False
        print(f"\n\thandmade_dist_mask: {handmade_dist_mask}")

        static_np = torch.from_numpy(static_np).to(self.config.device).float()
        _dynamic_np = torch.from_numpy(_dynamic_np).to(self.config.device).float()
        last_dim = torch.zeros((input_size, 2), dtype=static_np.dtype, device=static_np.device)
        last_dim = inst.get_last_dim(static_np, origin_idx)
        print(f"\n\tDEBUG: last_dim:")
        for el in last_dim:
            print('\t', el)
        dynamic_input_last_dim = torch.cat((_dynamic_np, last_dim), dim=-1)

        travel_time_norm = (dynamic_input_last_dim[:, -1] * inst.speed_f)/inst.max_time
        print(f"\n\tDEBUG: dynamic_input_last_dim:")
        for el in dynamic_input_last_dim:
            print('\t', el)
        print(f"\n\tDEBUG: dynamic_input_last_dim.shape: {dynamic_input_last_dim.shape}")
        print(f"\n\tDEBUG: travel_time_norm:")
        for el in travel_time_norm:
            print('\t', el)
        print(f"\n\tDEBUG: travel_time_norm.shape: {travel_time_norm.shape}")
        print(f"\n\tDEBUG: static_np:")
        for el in static_np:
            print('\t', el)
        print(f"\n\tDEBUG: static_np.shape: {static_np.shape}")

        bw_mask = get_backward_mask(
                time            = dynamic_input_last_dim[origin_idx][-1].item(),
                travel_times    = travel_time_norm,
                inst            = inst,
                tw_open         = static_np[:, 2])

        print(f"\nDEBUG: bw_mask: {bw_mask}")
        for i in range(len(bw_mask)):
            print(f"DEBUG: bw_mask[i] = {bw_mask[i]} | handmade_dist_mask[i] = {handmade_dist_mask[i]}")
            self.assertEqual(bw_mask[i], handmade_dist_mask[i])


        print(f"... finished test_get_backward_mask()")

    def test_get_mask(self):
        print(f"\nTesting test_get_mask...")
        inst = self.instance

        origin_idx = 2
        print(f"\nChosing origin_idx: {origin_idx}")
        input_size = inst.get_max_nb_input_points()
        static_np, _dynamic_np = inst.get_network_input(input_size)
        print(f"\nDEBUG: inst.nn_input_idx_to_tour:")
        for j, el in enumerate(inst.nn_input_idx_to_tour):
            print(j, el)
        print(f"\norigin_idx={origin_idx} corresponds to open tour:")
        print(inst.nn_input_idx_to_tour[origin_idx][0])
        print(f"\nSchedule with incomplete tour of nn_input at index {origin_idx}:")
        sched = inst.schedule[inst.solution.index(inst.nn_input_idx_to_tour[origin_idx][0])]
        pos = inst.nn_input_idx_to_tour[origin_idx][1]
        print(sched) 
        if pos == 0:
            time = sched[pos][0] 
        else:
            time = sched[pos][1]
        print(f"\nThe tour end has time:", time) 
        print(f"\ntime_windows:")
        for j, el in enumerate(inst.time_window):
            print(j, el)
        custs_to_test = []
        current_tour, current_pos = inst.nn_input_idx_to_tour[origin_idx]
        current_cust = current_tour[current_pos][0]
        for t in inst.nn_input_idx_to_tour:
            tour, pos = t
            cust = tour[pos][0]
            custs_to_test.append(cust)
        handmade_dist_mask = torch.ones(len(custs_to_test), dtype=bool, device=self.config.device)
        for j, c in enumerate(custs_to_test):
            tw_open, tw_close = inst.time_window[c]
            travel_time = inst.distances[current_cust][c]*inst.speed_f
            print(f"DEBUG: travel_time from {current_cust} to {c}: {travel_time}")
            if current_pos == 0: # then candidates should go before current route
                # check backwards
                backward_arrival = time - travel_time - inst.service_time
                print(f"DEBUG: backward_arrival = {backward_arrival} vs tw_open = {tw_open}")
                if tw_open <= backward_arrival:
                    handmade_dist_mask[j] = True
                else:
                    handmade_dist_mask[j] = False
            else:
                # check afterwards
                forward_arrival = time + travel_time + inst.service_time
                print(f"DEBUG: forward_arrival = {forward_arrival} vs tw_close = {tw_close}")
                if forward_arrival <= tw_close:
                    handmade_dist_mask[j] = True
                else:
                    handmade_dist_mask[j] = False

            if j == origin_idx:
                handmade_dist_mask[j] = False
        print(f"\nDEBUG: handmade_dist_mask: {handmade_dist_mask}")

        print(f"\nDEBUG: custs_to_test: {custs_to_test}")

        static_np = torch.from_numpy(static_np).to(self.config.device).float()
        _dynamic_np = torch.from_numpy(_dynamic_np).to(self.config.device).float()
        last_dim = torch.zeros((input_size, 2), dtype=static_np.dtype, device=static_np.device)
        last_dim = inst.get_last_dim(static_np, origin_idx)
        dynamic_input_last_dim = torch.cat((_dynamic_np, last_dim), dim=-1)
        static_np = static_np.unsqueeze(0)
        dynamic_input_last_dim = dynamic_input_last_dim.unsqueeze(0)

        mask = get_mask(
                origin_nn_input_idx = torch.tensor([origin_idx]),
                static_input        = static_np,
                dynamic_input       = dynamic_input_last_dim,
                instances           = [inst],
                config              = self.config,
                capacity            = inst.capacity)
        print(f"\n\tDEBUG: mask: {mask}")
        for i in range(len(mask[0])):
            print('\t', f"DEBUG: mask[0][i] = {mask[0][i]} | handmade_dist_mask[i] = {handmade_dist_mask[i]}")
            self.assertEqual(mask[0][i], handmade_dist_mask[i])

#        sol = vrp_to_plot_solution(inst)
#        data = pyvrp_read('./test/test_instance.vrp') 
#        plot_solution(sol, data, name='test_instance', path='/home/pettepiero/tirocinio/NLNS_cvrptw/temp/', plot_title=f'instance', plot_clients=True)
#

if __name__ == '__main__':
    unittest.main()

