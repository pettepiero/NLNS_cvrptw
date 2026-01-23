import unittest
from vrp.vrp_problem import VRPInstance, get_mask, get_backward_mask, get_forward_mask
from vrp.data_utils import read_instance_vrp, vrp_to_plot_solution
from pyvrp import read as pyvrp_read
from pyvrp.plotting import plot_solution
import numpy as np
import torch
from unittest.mock import Mock

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
        print(f"\n\tInitial solution:")
        for el in inst.solution:
            print('\t', el)
        print(f"\t...done")
        print(f"\n... finished setUp()")

    def test_get_forward_mask(self):
        print(f"\nTesting test_get_forward_mask()...")
        inst = self.instance
        print(f"\n\t Removing customer 3 and customer 9")
        inst.destroy([3, 9])  # remove customer 3 and 9 (pick any valid customer index)
        #inst.destroy([9])  # remove customer 9 (pick any valid customer index)
        print(f"\n\tAfter destruction:")
        for el in inst.solution:
            print('\t', el)
        print(f"\n\tAfter schedule:")
        for el in inst.schedule:
            print('\t', el)
        print(f"\n\tAfter destruction, incomplete tours:")
        for el in inst.incomplete_tours:
            print('\t', el)

        origin_idx = 1
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
                # check forwards
                forward_arrival = round(time - travel_time - inst.service_time, 2)
                if tw_open <= forward_arrival:
                    handmade_dist_mask[j] = True
                    print(f"\tDEBUG: forward_arrival = {forward_arrival} vs tw_open = {tw_open}: \tFeasible")
                else:
                    handmade_dist_mask[j] = False
                    print(f"\tDEBUG: forward_arrival = {forward_arrival} vs tw_open = {tw_open}: \tUnfeasible")
            else:
                # check afterwards
                forward_arrival = round(time + travel_time + inst.service_time, 2)
                if forward_arrival <= tw_close:
                    handmade_dist_mask[j] = True
                    print(f"\tDEBUG: forward_arrival = {forward_arrival} vs tw_close = {tw_close}: \tFeasible")
                else:
                    handmade_dist_mask[j] = False
                    print(f"\tDEBUG: forward_arrival = {forward_arrival} vs tw_close = {tw_close}: \tUnfeasible")

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

        fw_mask = get_forward_mask(
                origin_idx      = origin_idx,
                time            = dynamic_input_last_dim[origin_idx][-1].item(),
                travel_times    = travel_time_norm,
                inst            = inst,
                tw_close        = static_np[:, 3])

        print(f"\n\tDEBUG: fw_mask: {fw_mask}")
        for i in range(len(fw_mask)):
            print(f"\tDEBUG: fw_mask[i] = {fw_mask[i]} | handmade_dist_mask[i] = {handmade_dist_mask[i]}")
            self.assertEqual(fw_mask[i], handmade_dist_mask[i])

        print(f"... finished test_get_forward_mask()")

    def test_get_backward_mask(self):
        print(f"\nTesting test_get_backward_mask()...")
        inst = self.instance
        print(f"\n\t Removing customer 5")
        inst.destroy([5])  # remove customer 5 (pick any valid customer index)
        print(f"\n\tAfter destruction:")
        for el in inst.solution:
            print('\t', el)
        print(f"\n\tAfter schedule:")
        for el in inst.schedule:
            print('\t', el)
        
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
                origin_idx      = origin_idx,
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

