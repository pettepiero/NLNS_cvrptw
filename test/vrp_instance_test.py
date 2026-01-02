import unittest
from vrp.vrp_problem import VRPInstance
from vrp.data_utils import read_instance_vrp
import numpy as np

class test_get_costs(unittest.TestCase):
    def setUp(self):
        self.instance = read_instance_vrp('./test/test_instance.vrp')

    def test_get_cost_complete(self):
        inst = self.instance
        self.assertEqual(inst.late_coeff, 1) 
        inst.create_initial_solution() 
        #print("Instance initial solution:")
        #for el in inst.solution:
        #    print(el)
        #print(f"Instance distances:")
        #for i, el in enumerate(inst.distances):
        #    print(f"From {i}")
        #    for j, el2 in enumerate(el):
        #        print(f"To {j}: {el2}")

        #print("Instance schedule: ")
        #for el in inst.schedule:
        #    print(el)
        #print(f"Instance time window: ")
        #for i, el in enumerate(inst.time_window):
        #    print(i, el)

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

        self.assertEqual(delay, 2885) # compare with hand computed delay

        # Computing total distance of solution
        dist = 0
        for route_idx, route in enumerate(inst.solution):
            for el_idx, el in enumerate(route[:-1]):
                cust = el[0]
                next_cust = route[el_idx+1][0]
                distance = np.sqrt((inst.original_locations[cust, 0] - inst.original_locations[next_cust, 0]) ** 2
                             + (inst.original_locations[cust, 1] - inst.original_locations[next_cust, 1]) ** 2)
                #distance = inst.distances[cust][next_cust]
                dist += distance

        self.assertTrue(abs(dist-5398.025) <= 0.1)
        total_cost = dist + inst.late_coeff*delay
        print(f"total_cost: {total_cost} = {dist} + {inst.late_coeff}*{delay}")
        print(f"inst.get_costs(True): {inst.get_costs(True)}")
        self.assertTrue(abs(total_cost - inst.get_costs(False)) <= 0.001)
        self.assertTrue(abs(total_cost - inst.get_costs(True)) <= 1)
    
    def test_get_sum_late_mins(self):
        inst = self.instance
        inst.create_initial_solution() 
        self.assertEqual(inst.get_sum_late_mins(), 2885)

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

if __name__ == '__main__':
    unittest.main()

