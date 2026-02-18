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

        known_init_sol = [
            [[0, 0, 0]],
            [[0, 0, 0], [2, 2, None], [3, 8, None], [0, 0, 0]],
            [[0, 0, 0], [5, 6, None], [10, 4, None], [7, 4, None], [0, 0, 0]],
            [[0, 0, 0], [8, 8, None], [4, 5, None], [0, 0, 0]],
            [[0, 0, 0], [6, 10, None], [9, 7, None], [1, 3, None], [0, 0, 0]]
            ]
        for kn, sol in zip(known_init_sol, inst.solution):
            assert kn == sol
        del known_init_sol

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
        out = inst.get_last_dim(static_input, origin_idx, print_debug=True)

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
        print(f"DEBUG: tours:")
        for j, el in enumerate(tours):
            print(j, el)
        print(f"DEBUG: solution:")
        for j, el in enumerate(inst.solution):
            print(j, el)
        print(f"DEBUG: schedule:")
        for j, el in enumerate(inst.schedule):
            print(j, el)

        #indices = list(map(inst.solution.index, tours))
        indices = list(map(inst.get_idx_in_solution, tours, pos_in_tours))

        schedules = [inst.schedule[ind] for ind in indices]
        expected_times = []
        for idx, el in enumerate(zip(tours, range(len(indices)))):
            tour, j = el
            print(f"DEBUG: doing tour: {tour}")
            if len(tour) == 1:
                if idx == 0:
                    expected_times.append(schedules[j][pos_in_tours[j]][1]/inst.max_time)
                else:
                    if tours[idx -1][0][0] == tour[0][0]:
                        print(f"DEBUG: second time with customer {tour[0][0]}")
                        expected_times.append(schedules[j][pos_in_tours[j]][0]/inst.max_time)
                    else:
                        print(f"DEBUG: first time with customer {tour[0][0]}")
                        expected_times.append(schedules[j][pos_in_tours[j]][1]/inst.max_time)
            elif pos_in_tours[j] == 0:
                expected_times.append(schedules[j][pos_in_tours[j]][0]/inst.max_time)
            else:
                expected_times[j].append(schedules[j][pos_in_tours[j]][1]/inst.max_time)
        expected_times = torch.tensor(expected_times, dtype=expected_dists.dtype, device=expected_dists.device)
        print(f"DEBUG: expected_times: {expected_times}")
        print(f"DEBUG: expected_dists: {expected_dists}")

        expected = torch.stack((expected_dists, expected_times), dim=1)
        print(f"DEBUG: expected: {expected}")
        print(f"out_1d: {out_1d}")

        # 4) compare
        self.assertTrue(torch.allclose(out_1d, expected, atol=1e-6))


class test_get_last_dim2(unittest.TestCase):
    def setUp(self):
        self.instance = read_instance_vrp('./test/test_instance.vrp')

    def test_get_last_dim_values(self):
        inst = self.instance
        inst.create_initial_solution()

        known_init_sol = [
            [[0, 0, 0]],
            [[0, 0, 0], [2, 2, None], [3, 8, None], [0, 0, 0]],
            [[0, 0, 0], [5, 6, None], [10, 4, None], [7, 4, None], [0, 0, 0]],
            [[0, 0, 0], [8, 8, None], [4, 5, None], [0, 0, 0]],
            [[0, 0, 0], [6, 10, None], [9, 7, None], [1, 3, None], [0, 0, 0]]
            ]
        for kn, sol in zip(known_init_sol, inst.solution):
            assert kn == sol
        del known_init_sol

        # Make the instance "incomplete" so get_network_input() builds nn_input_idx_to_tour
        print(f"Initial solution:")
        for el in inst.solution:
            print(el)

        print(f"\n removing customer 8, which has coordinates: {inst.locations[8]}")
        print(f"\n removing customer 3, which has coordinates: {inst.locations[3]}")
        # and open_nn_input_idx, which get_last_dim relies on.
        inst.destroy([8, 3])  # remove customers 8 and 3
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
        out = inst.get_last_dim(static_input, origin_idx, print_debug=True)

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
        print(f"DEBUG: tours:")
        for j, el in enumerate(tours):
            print(j, el)
        print(f"DEBUG: solution:")
        for j, el in enumerate(inst.solution):
            print(j, el)
        print(f"DEBUG: schedule:")
        for j, el in enumerate(inst.schedule):
            print(j, el)

        #indices = list(map(inst.solution.index, tours))
        indices = list(map(inst.get_idx_in_solution, tours, pos_in_tours))

        schedules = [inst.schedule[ind] for ind in indices]
        expected_times = []
        for idx, el in enumerate(zip(tours, range(len(indices)))):
            tour, j = el
            print(f"DEBUG: doing tour: {tour}")
            if len(tour) == 1:
                if idx == 0:
                    expected_times.append(schedules[j][pos_in_tours[j]][1]/inst.max_time)
                else:
                    if tours[idx -1][0][0] == tour[0][0]:
                        print(f"DEBUG: second time with customer {tour[0][0]}")
                        expected_times.append(schedules[j][pos_in_tours[j]][0]/inst.max_time)
                    else:
                        print(f"DEBUG: first time with customer {tour[0][0]}")
                        expected_times.append(schedules[j][pos_in_tours[j]][1]/inst.max_time)
            elif pos_in_tours[j] == 0:
                expected_times.append(schedules[j][pos_in_tours[j]][0]/inst.max_time)
            else:
                expected_times.append(schedules[j][pos_in_tours[j]][1]/inst.max_time)
        expected_times = torch.tensor(expected_times, dtype=expected_dists.dtype, device=expected_dists.device)
        print(f"DEBUG: expected_times: {expected_times}")
        print(f"DEBUG: expected_dists: {expected_dists}")

        expected = torch.stack((expected_dists, expected_times), dim=1)
        print(f"DEBUG: expected: {expected}")
        print(f"out_1d: {out_1d}")

        # 4) compare
        self.assertTrue(torch.allclose(out_1d, expected, atol=1e-6))



class test_schedule_functions(unittest.TestCase):
    def setUp(self):
        self.instance = read_instance_vrp('./test/test_instance.vrp')
        inst = self.instance
        inst.create_initial_solution()

        expected_initial_tours = [
                [[0, 0, 0]],
                [[0, 0, 0], [2, 2, None], [3, 8, None], [0, 0, 0]],
                [[0, 0, 0], [5, 6, None], [10, 4, None], [7, 4, None], [0, 0, 0]],
                [[0, 0, 0], [8, 8, None], [4, 5, None], [0, 0, 0]],
                [[0, 0, 0], [6, 10, None], [9, 7, None], [1, 3, None], [0, 0, 0]]]

        for j, el in enumerate(inst.solution):
            assert el == expected_initial_tours[j]

        inst.destroy([5, 6, 3])  # remove customer 5 (pick any valid customer index)
        
        expected_after_destruction = [
                [[0, 0, 0]],
                [[0, 0, 0], [2, 2, None]],
                [[3, 8, None]],
                [[5, 6, None]],
                [[10, 4, None], [7, 4, None], [0, 0, 0]],
                [[0, 0, 0], [8, 8, None], [4, 5, None], [0, 0, 0]],
                [[6, 10, None]],
                [[9, 7, None], [1, 3, None], [0, 0, 0]]]

        for j, el in enumerate(inst.solution):
            assert el == expected_after_destruction[j]
    
    def test_get_tour_end_position(self):
        inst = self.instance
        self.assertEqual(inst.get_tour_end_position(inst.solution[1]), 1)
        self.assertEqual(inst.get_tour_end_position(inst.solution[2]), 0)
        self.assertEqual(inst.get_tour_end_position(inst.solution[4]), 0)
        self.assertEqual(inst.get_tour_end_position(inst.solution[7]), 0)
        

    def test_get_schedule_for_backw_ins(self):
        inst = self.instance
        #print("\n\n")
        #print(f"TIME WINDOWS:")
        #for j, el in enumerate(inst.time_window):
        #    print(j, el)
        #print(f"DISTANCES:")
        #for j, el in enumerate(inst.distances):
        #    print(j, el)
        #print(f"inst.speed_f = {inst.speed_f} | service_time = {inst.service_time}")
        #print(f"tour: {inst.solution[4]}")
        sched = inst.get_schedule_for_backw_ins(inst.solution[4])

        expected_sched = [[7027, 7127], [8187, 8287], [10000, 10000]]
        for exp, comp in zip(expected_sched, sched):
            self.assertEqual(exp, comp)

        #print(f"tour: {inst.solution[7]}")
        expected_sched = [[2779, 2879], [4334, 4434], [10000, 10000]]
        sched = inst.get_schedule_for_backw_ins(inst.solution[7])
        for exp, comp in zip(expected_sched, sched):
            self.assertEqual(exp, comp)

        self.instance = read_instance_vrp('./test/test_instance.vrp')
        inst = self.instance
        inst.create_initial_solution()
        inst.destroy([1])  # remove customer 5 (pick any valid customer index)
        inst.solution[-2] = inst.solution[-2][1:]
        print("\n")
        for el in inst.solution:
            print(el)

        #print(f"tour: {inst.solution[-2]}")
        expected_sched = [[1635, 1735], [2779, 2879]]
        sched = inst.get_schedule_for_backw_ins(inst.solution[-2])
        for exp, comp in zip(expected_sched, sched):
            self.assertEqual(exp, comp)

        #print(f"TIME WINDOWS:")
        #for j, el in enumerate(inst.time_window):
        #    print(j, el)
        #print(f"DISTANCES:")
        #for j, el in enumerate(inst.distances):
        #    print(j, el)
        #print(f"inst.speed_f = {inst.speed_f} | service_time = {inst.service_time}")
        #print(f"tour: {inst.solution[-1]}")

        expected_sched = [[4334, 4434]]
        sched = inst.get_schedule_for_backw_ins(inst.solution[-1])
        for exp, comp in zip(expected_sched, sched):
            self.assertEqual(exp, comp)

    def test_get_schedule_for_forw_ins(self):
        inst = self.instance
        #print("Instance initial solution:")
        #for el in inst.solution:
        #    print(el)
        #print("\n\n")
        #print(f"TIME WINDOWS:")
        #for j, el in enumerate(inst.time_window):
        #    print(j, el)
        #print(f"DISTANCES:")
        #for j, el in enumerate(inst.distances):
        #    print(j, el)
        #print(f"inst.speed_f = {inst.speed_f} | service_time = {inst.service_time}")
        #print(f"tour: {inst.solution[1]}")
        sched = inst.get_schedule_for_forw_ins(inst.solution[1])

        expected_sched = [[0, 0], [8192, 8292]]
        for exp, comp in zip(expected_sched, sched):
            self.assertEqual(exp, comp)

        self.instance = read_instance_vrp('./test/test_instance.vrp')
        inst = self.instance
        inst.create_initial_solution()
        inst.destroy([7])  # remove customer 5 (pick any valid customer index)

        sched = inst.get_schedule_for_forw_ins(inst.solution[2])
        expected_sched = [[0, 0], [2410, 2510], [5087, 5187]]
        for exp, comp in zip(expected_sched, sched):
            self.assertEqual(exp, comp)


        inst.solution[-2] = inst.solution[-2][1:-1]
        sched = inst.get_schedule_for_forw_ins(inst.solution[-2])
        expected_sched = [[495, 595], [4402, 4502]]
        for exp, comp in zip(expected_sched, sched):
            self.assertTrue(exp[0] - comp[0] < 1)
            self.assertTrue(exp[1] - comp[1] < 1)

if __name__ == '__main__':
    unittest.main()

