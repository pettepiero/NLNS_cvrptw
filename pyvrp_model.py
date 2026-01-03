# NOTE: if this code fails for unhashable numpy type, add following line in vrplib parse_section function:

#        if name == "vehicles_depot":
#            data = np.array([row[0] for row in rows])
#        else:
#            data = np.array([row[1:] for row in rows])


import argparse
from pyvrp import Model, read, Result, solve
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution, plot_coordinates, plot_route_schedule
import matplotlib.pyplot as plt
import numpy as np
from vrplib.read import read_instance
from pathlib import Path
from vrp.vrp_problem import VRPInstance
import os
from typing import Union
from tqdm import tqdm
import logging
from vrp.logger import setup_logging
import math 

setup_logging(log_file="logs/pyvrp.log", level=logging.DEBUG, to_console=False)

log = logging.getLogger(__name__)

def run_pyvrp_on_instance(inst: VRPInstance, only_cost: bool = False, display: bool = False) -> Union[float, tuple]:
    """
    Runs PyVRP on instance. Returns only cost is only_cost is True, otherwise returns Result.
    """
    log.info(f'In run_pyvrp_on_instance | instance: {inst}')
    m = Model()
    num_depots = len(inst['depot'])
    log.info(f'In run_pyvrp_on_instance | num_depots: {num_depots}')
    depots = []
    for d in inst['depot']:
        depot = m.add_depot(x=inst['node_coord'][d][0], y=inst['node_coord'][d][1])
        depots.append(depot)
    log.info(f'In run_pyvrp_on_instance | depots: {depots}')
    
    if 'vehicles' in list(inst.keys()) and str(inst['vehicles']) != 'inf':
        keys, counts = np.unique(inst['vehicles_depot'], return_counts=True)
        keys = keys - 1
        keys = keys.tolist()
        counts = counts.tolist()
        depot_num_vehicles =  dict(zip(keys, counts))
    else:
        depot_num_vehicles = {}
        for i, d in enumerate(inst['depot']):
            depot_num_vehicles[i] = inst['dimension'] - num_depots
    log.info(f'In run_pyvrp_on_instance | depot_num_vehicles: {depot_num_vehicles}')
    
    for i, d in enumerate(inst['depot']):
        m.add_vehicle_type(
            num_available   = depot_num_vehicles[int(d)],
            capacity        = inst['capacity'],
            start_depot     = depots[i],
            end_depot       = depots[i],
        )
    
    clients = [
        m.add_client(
            x=int(inst['node_coord'][idx][0]),
            y=int(inst['node_coord'][idx][1]),
            delivery=int(inst['demand'][idx]),
        )
        for idx in range(num_depots, len(inst['node_coord']))
    ]
    
    locations = depots + clients
    
    for frm_idx, frm in enumerate(locations):
        for to_idx, to in enumerate(locations):
            #distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
            distance = np.sqrt((frm.x - to.x)**2 + (frm.y - to.y)**2) 
            m.add_edge(frm, to, distance=distance)
    
    result = m.solve(stop=MaxRuntime(args.max_time), display=display)
    print(f"Result: {result}")
    
    if only_cost:    
        return result.best.distance_cost() 
    else:
        return result, m

def read_instances(dir_path: Path) -> list:
    list_of_files = os.listdir(dir_path)
    log.info(f'list_of_files: {list_of_files}')
    cwd = os.getcwd()
    instances = []
    for inst in list_of_files:
        data_path = os.path.join(cwd, dir_path, inst)
        if os.path.splitext(data_path)[1] != '.mdvrp' and os.path.splitext(data_path)[1] != '.vrp':
            continue
        data = read_instance(data_path)
        instances.append(data)
    return instances

def eval_single(args):
    data = read(args.instance_path)
    parent_dir = Path(args.instance_path).parent.absolute()
    instance = read_instances(parent_dir)[0]
    result, m = run_pyvrp_on_instance(
        inst        = instance,
        only_cost   = False,
        display     = True
        )
    cost = result.best.distance_cost()

    output_filename = os.path.join(args.output_dir, "search", "pyvrp_eval.txt" ) 
    with open(output_filename, 'a') as f:
        f.write(f"0,{cost}\n")

    print(f"Written cost results of PyVRP in file: {output_filename}")



    #result = solve(data, stop=MaxRuntime(args.max_time), display=True)
    #results = [[0, result.best.distance_cost()]]
    #print(f"DEBUG: results: {results}")
    #
    #output_filename = os.path.join(args.output_dir, "search", "pyvrp_eval_batch.txt" ) 
    #with open(output_filename, 'a') as f:
    #    for el in results:
    #        print(el)
    #        i, cost = el
    #        f.write(f"{i},{cost}\n")
    #print(f"Written cost results of PyVRP in file: {output_filename}")

    if args.plot_solution:
        #figures_path = Path('/home/pettena/NLNSTW/temp/')
        figures_path = Path('/home/pettepiero/tirocinio/NLNS_cvrptw/temp/')
        _, ax = plt.subplots(figsize=(8, 8))
        plot_solution(result.best, data, path=figures_path, name="pyvrp_final_sol", plot_title='PyVRP final sol')
    


def eval_batch(args):
    log.info('Passed batch path: {args.dir_path}')
    instances = read_instances(args.dir_path)
    log.info(f'Read instances: {instances}')
    results = [] 
    for i, inst in enumerate(tqdm(instances)):
        cost = run_pyvrp_on_instance(
                inst        = inst,
                only_cost   = True,
                display     = False)
        results.append([i, cost])

    output_filename = os.path.join(args.output_dir, "search", "pyvrp_eval.txt" ) 
    with open(output_filename, 'a') as f:
        for el in results:
            print(el)
            i, cost = el
            f.write(f"{i},{cost}\n")

    print(f"Written cost results of PyVRP in file: {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyVRP model execution")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['eval_single', 'eval_batch'],
        required=True,
        help="Mode of operation. Valid choices are 'eval_single' and 'eval_batch'"
    )
    parser.add_argument( 
        "--instance_path",
        default=None,
        type=str,
        help="Path to one VRPLIB file",
    )
    parser.add_argument( 
        "--dir_path",
        default=None,
        type=str,
        help="Path to dir with VRPLIB files",
    )
    parser.add_argument( 
        "--output_dir",
        default=None,
        type=str,
        help="Path to dir where results should be solved",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=10.0,
        help="Max runtime per instance in seconds (float).",
    )
    parser.add_argument(
        "--plot_solution", "--plot-solution",
        dest='plot_solution',
        type=bool,
        default=False,
        help="Plot instance solution.",
    )
    args = parser.parse_args()
    log.info('In pyvrp_model.py')    
    if args.mode == 'eval_batch':
        log.info('Chose eval_batch mode')
        assert args.dir_path is not None, f"Missing argument dir_path in 'eval_batch' mode"
        assert args.output_dir is not None, f"Missing argument output_dir in 'eval_batch' mode"

        eval_batch(args)
    
    elif args.mode == 'eval_single':
        log.info('Chose eval_single mode')
        assert args.instance_path is not None, f"Missing argument instance_path in 'eval_single' mode"
        eval_single(args)
