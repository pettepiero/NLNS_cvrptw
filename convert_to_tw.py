import argparse
from pyvrp import Model, read, Result, solve
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution, plot_coordinates, plot_route_schedule
import matplotlib.pyplot as plt
import numpy as np
from vrplib.read import read_instance
from vrplib import write_instance
from pathlib import Path
from vrp.vrp_problem import VRPInstance
import os
import shutil
from os import listdir
from os.path import isfile, exists, isdir, join, splitext, split
from typing import Union
from tqdm import tqdm
import logging
from vrp.logger import setup_logging
import math 
from statistics import median

setup_logging(log_file="logs/pyvrp.log", level=logging.DEBUG, to_console=False)

log = logging.getLogger(__name__)

def get_result_max_dist(result, data):
    duration_matrix = data.duration_matrices()[0]
    max_dist = 0
    for r in result:
        dist = 0
        for i in range(len(r)-1):
            cust        = r[i]
            next_cust   = r[i+1]
            service_duration = int(data.clients()[cust-1].service_duration)
            dist += int(duration_matrix[cust][next_cust])
            dist += service_duration
        if dist > max_dist:
            max_dist = dist
    return max_dist

def get_dist_median(data):
    duration_matrix = data.duration_matrices()[0]
    d_list = [l for ll in duration_matrix for l in ll]
    return int(median(d_list))

def sample_slacks_beta(centre, horizon, hmin, hmax, rng, alpha=2.0, beta=6.0):
    u_left = rng.beta(alpha, beta)
    u_right = rng.beta(alpha, beta)

    l = hmin + u_left*(hmax-hmin)
    r = hmin + u_right*(hmax-hmin)
    L = int(max(0, centre - l))
    R = int(min(horizon, centre + r))

    return L, R, l, r


def get_time_windows_from_instance(instance, H, delta, hmin, hmax, stop_criterion, rng):
    data = read(instance)
    result = solve(data, stop=stop_criterion, display=False)
    solution = result.best
    routes = solution.routes()
    dist_median = get_dist_median(data)
    max_dist = get_result_max_dist(routes, data) 

    if max_dist <= H - delta:
        # then the instance is consired valid for cvrptw
        # append depot to routes to first and last position
        routes_lists = []
        for r in routes:
            rl = list(r)
            rl.insert(0, 0)
            rl.insert(len(rl), 0)
            routes_lists.append(rl)

        time_windows = [[0, H] for l in range(data.num_clients + data.num_depots)]
        for r in routes_lists:
            prev_c = 0
            sched = []
            for i, cust in enumerate(r[:-1]): 
                if i == 0:
                    continue
                if i == 1:
                    #sample start time
                    c = rng.integers(low=0, high=H, size=1)
                else:
                    prev_cust = r[i-1]
                    service_time = int(data.clients()[prev_cust-1].service_duration)
                    dist = int(data.duration_matrices()[0][prev_cust][cust])
                    c = prev_c + dist
                left, right, _, _ = sample_slacks_beta(c, H, hmin, hmax, rng)
                time_windows[cust] = [left, right]
                prev_c = c 
    else:
        # then the instance is not considered valid for cvrptw
        print(f"\nInstance {instance} could not be converted to CVRPTW because the longest route is not shorter than H - delta = {H-delta}")
        print(f"Skipping instance {instance}\n")

    return time_windows

def fix_dict(d):
    d['TIME_WINDOW_SECTION'] = time_windows
    # capitalize keys
    d = {k.upper(): v for k, v in d.items()}
    # drop EDGE_WEIGHT
    d.pop('EDGE_WEIGHT', None)
    # drop depot section
    d.pop('DEPOT', None)
    # add depot_section
    d['DEPOT_SECTION'] = [1, -1]
    # change type to CVRPTW
    d['TYPE'] = 'CVRPTW'
    # add service time
    d['SERVICE_TIME'] = 10
    # change NODE_COORD to NODE_COORD_SECTION
    d['NODE_COORD_SECTION'] = d.pop('NODE_COORD')
    # change DEMAND to DEMAND_SECTION
    d['DEMAND_SECTION'] = d.pop('DEMAND')
    # change order of dictionary
    final_dict = reorder_dict(d)

    return final_dict
    


def reorder_dict(d):
    ordered_keys = ['NAME', 'TYPE', 'DIMENSION', 'CAPACITY', 'SERVICE_TIME',
                    'EDGE_WEIGHT_TYPE', 'NODE_COORD_SECTION', 'DEMAND_SECTION',
                    'TIME_WINDOW_SECTION', 'DEPOT_SECTION']

    new_d = {k: d[k] for k in ordered_keys}
    return new_d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts CVRP instances to CVRPTW")
    parser.add_argument( 
        "--dir_path",
        type=str,
        help="Path to dir with VRPLIB files",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=30.0,
        help="Max runtime per instance in seconds (float).",
    )
    parser.add_argument(
        "--horizon", "-H",
        type=int,
        default=10000,
        help="Time horizon of problems"
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=3000,
        help="Time delta for valid solutions"
    )

    args = parser.parse_args()
    log.info('In convert_to_tw.py')    

    #time horizon and delta
    H = args.horizon
    delta = args.delta

    hmin = 100
    hmax = 5000

    # set up random generator
    rng = np.random.default_rng()


    # get instances
    assert exists(args.dir_path) & isdir(args.dir_path)
    instances = [f for f in listdir(args.dir_path) if isfile(join(args.dir_path, f))]
    instances = [join(args.dir_path, f) for f in instances]

    assert len(instances) > 0, f"Found no instances in {args.dir_path}"
    new_folder = Path(args.dir_path + '_tw')
    print(f"Instances will be saved to: {new_folder}")
    if exists(new_folder):
        shutil.rmtree(new_folder)
    new_folder.mkdir()

    # get solutions
    for instance in tqdm(instances):
        stop_criterion = MaxRuntime(args.max_time)
        time_windows = get_time_windows_from_instance(instance, H, delta, hmin, hmax, stop_criterion, rng)
        old_name = splitext(instance)[0]
        old_name = split(old_name)[-1]
        new_instance_name = Path(new_folder, old_name + f'_tw' + '.vrp') 

        inst_vrplib = read_instance(instance)  
        final_dict = fix_dict(inst_vrplib)
        write_instance(new_instance_name, final_dict)
        print(f"Converted instance {instance} to {new_instance_name}")
