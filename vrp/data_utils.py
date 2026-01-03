import numpy as np
from vrp.vrp_problem import VRPInstance
import pickle
from tqdm import trange, tqdm
from cordeau_read_cvrptw import convert_cordeau_to_vrplib
import os
from typing import Iterable, List

class InstanceBlueprint:
    """Describes the properties of a certain instance type (e.g. number of customers)."""

    def __init__(self, nb_customers, depot_position, customer_position, nb_customer_cluster, demand_type, demand_min,
                 demand_max, capacity, grid_size, service_time, late_coeff, max_time):
        self.nb_customers           = nb_customers
        self.depot_position         = depot_position
        self.customer_position      = customer_position
        self.nb_customers_cluster   = nb_customer_cluster
        self.demand_type            = demand_type
        self.demand_min             = demand_min
        self.demand_max             = demand_max
        self.capacity               = capacity
        self.grid_size              = grid_size
        self.service_time           = service_time
        #self.early_coeff           = early_coeff
        self.late_coeff             = late_coeff
        self.max_time               = max_time


def get_blueprint(blueprint_name):
    type = blueprint_name.split('_')[0]
    instance = blueprint_name.split('_')[1]
    if type == "ALTR":
        import vrp.dataset_blueprints.ALTR
        return vrp.dataset_blueprints.ALTR.dataset[instance]
    elif type == "XE":
        import vrp.dataset_blueprints.XE
        return vrp.dataset_blueprints.XE.dataset[instance]
    elif type == "S":
        import vrp.dataset_blueprints.S
        return vrp.dataset_blueprints.S.dataset[instance]
    elif type == "TW":
        import vrp.dataset_blueprints.TW
        return vrp.dataset_blueprints.TW.dataset[instance]
    raise Exception('Unknown blueprint instance')


def create_dataset(size, config, rng, create_solution=False, use_cost_memory=True):
    instances = []
    blueprints = get_blueprint(config.instance_blueprint)

    for i in trange(size):
        if isinstance(blueprints, list):
            blueprint_rnd_idx = rng.integers(0, len(blueprints), 1).item()
            vrp_instance = generate_Instance(blueprints[blueprint_rnd_idx], use_cost_memory, rng)
        else:
            vrp_instance = generate_Instance(blueprints, use_cost_memory, rng)
        instances.append(vrp_instance)
        if create_solution:
            vrp_instance.create_initial_solution()
    return instances


def generate_Instance(blueprint, use_cost_memory, rng):
    depot_position      = get_depot_position(blueprint, rng)
    customer_position   = get_customer_position(blueprint, rng)
    demand              = get_customer_demand(blueprint, customer_position, rng)
    original_locations  = np.insert(customer_position, 0, depot_position, axis=0)
    demand              = np.insert(demand, 0, 0, axis=0)
    time_window         = get_time_window(blueprint, rng)

    if blueprint.grid_size == 1000:
        locations = original_locations / 1000
    elif blueprint.grid_size == 1000000:
        locations = original_locations / 1000000
    else:
        assert blueprint.grid_size == 1
        locations = original_locations

    vrp_instance = VRPInstance(
        nb_customers        = blueprint.nb_customers, 
        locations           = locations, 
        original_locations  = original_locations, 
        demand              = demand, 
        capacity            = blueprint.capacity,
        time_window         = time_window,
        service_time        = blueprint.service_time,
        #early_coeff         = blueprint.early_coeff,
        late_coeff          = blueprint.late_coeff,
        use_cost_memory     = use_cost_memory,
        max_time            = blueprint.max_time)
    return vrp_instance

def get_time_window(blueprint, rng):
    """
    Generate time windows following the description in https://www.jstor.org/stable/822953
    """
    max_time = blueprint.max_time 
    min_time = 0
    avg_gap = max_time/10

    centres         = rng.uniform(size=blueprint.nb_customers, low=min_time + avg_gap/2, high=max_time - avg_gap/2)
    windows_widths  = rng.normal(loc=avg_gap, scale=np.sqrt(avg_gap), size=blueprint.nb_customers)
    tw = [[max(c-w, min_time) , min(c+w, max_time)] for c,w in zip(centres, windows_widths)]
    tw.insert(0, [min_time, max_time])
    
    return np.array(tw, dtype=int)

def get_depot_position(blueprint, rng):
    if blueprint.depot_position == 'R':
        if blueprint.grid_size == 1:
            return rng.uniform(size=(1, 2))
        elif blueprint.grid_size == 1000:
            return rng.integers(0, 1001, 2)
        elif blueprint.grid_size == 1000000:
            return rng.integers(0, 1000001, 2)
    elif blueprint.depot_position == 'C':
        if blueprint.grid_size == 1:
            return np.array([0.5, 0.5])
        elif blueprint.grid_size == 1000:
            return np.array([500, 500])
    elif blueprint.depot_position == 'E':
        return np.array([0, 0])
    else:
        raise Exception("Unknown depot position")


def get_customer_position_clustered(nb_customers, blueprint, rng):
    assert blueprint.grid_size == 1000
    random_centers = rng.integers(0, 1001, (blueprint.nb_customers_cluster, 2))
    customer_positions = []
    while len(customer_positions) + blueprint.nb_customers_cluster < nb_customers:
        random_point = rng.integers(0, 1001, (1, 2))
        a = random_centers
        b = np.repeat(random_point, blueprint.nb_customers_cluster, axis=0)
        distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
        acceptance_prob = np.sum(np.exp(-distances / 40))
        if acceptance_prob > rng.random():
            customer_positions.append(random_point[0])
    return np.concatenate((random_centers, np.array(customer_positions)), axis=0)


def get_customer_position(blueprint, rng):
    if blueprint.customer_position == 'R':
        if blueprint.grid_size == 1:
            return rng.uniform(size=(blueprint.nb_customers, 2))
        elif blueprint.grid_size == 1000:
            return rng.integers(0, 1001, (blueprint.nb_customers, 2))
        elif blueprint.grid_size == 1000000:
            return rng.integers(0, 1000001, (blueprint.nb_customers, 2))
    elif blueprint.customer_position == 'C':
        return get_customer_position_clustered(blueprint.nb_customers, blueprint)
    elif blueprint.customer_position == 'RC':
        customer_position = get_customer_position_clustered(int(blueprint.nb_customers / 2), blueprint)
        customer_position_2 = rng.integers(0, 1001, (blueprint.nb_customers - len(customer_position), 2))
        return np.concatenate((customer_position, customer_position_2), axis=0)


def get_customer_demand(blueprint, customer_position, rng):
    if blueprint.demand_type == 'inter':
        return rng.integers(blueprint.demand_min, blueprint.demand_max + 1, size=blueprint.nb_customers)
    elif blueprint.demand_type == 'U':
        return np.ones(blueprint.nb_customers, dtype=int)
    elif blueprint.demand_type == 'SL':
        small_demands_nb = int(rng.uniform(0.7, 0.95, 1).item() * blueprint.nb_customers)
        demands_small = rng.integers(1, 11, size=small_demands_nb)
        demands_large = rng.integers(50, 101, size=blueprint.nb_customers - small_demands_nb)
        demands = np.concatenate((demands_small, demands_large), axis=0)
        np.random.shuffle(demands)
        return demands
    elif blueprint.demand_type == 'Q':
        assert blueprint.grid_size == 1000
        demands = np.zeros(blueprint.nb_customers, dtype=int)
        for i in range(blueprint.nb_customers):
            if (customer_position[i][0] > 500 and customer_position[i][1] > 500) or (
                    customer_position[i][0] < 500 and customer_position[i][1] < 500):
                demands[i] = rng.integers(51, 101, 1).item()
            else:
                demands[i] = rng.integers(1, 51, 1).item()
        return demands
    elif blueprint.demand_type == 'minOrMax':
        demands_small = np.repeat(blueprint.demand_min, blueprint.nb_customers * 0.5)
        demands_large = np.repeat(blueprint.demand_max, blueprint.nb_customers - (blueprint.nb_customers * 0.5))
        demands = np.concatenate((demands_small, demands_large), axis=0)
        np.random.shuffle(demands)
        return demands
    else:
        raise Exception("Unknown customer demand.")


def read_instance(path, pkl_instance_idx=0):
    if path.endswith('.vrp'):
        return read_instance_vrp(path)
    elif path.endswith('.sd'):
        return read_instance_sd(path)
    elif path.endswith('.pkl'):
        return read_instances_pkl(path, pkl_instance_idx, 1)[0]
    else:
        raise Exception("Unknown instance file type.")

def get_max_time(tw):
    max_t = max(tw[:,1])
    if 0 < max_t <= 100:
        return 100
    elif 100 < max_t <= 1000:
        return 1000
    elif 1000 < max_t <= 10000:
        return 10000
    else:
        raise ValueError
    

def read_instance_vrp(path):
    with open(path, 'r') as file:
        lines = [ll.strip() for ll in file]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith("SERVICE_TIME"):
            service_time = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('TIME_WINDOW_SECTION'):
            time_window = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
        elif line.startswith('LATE_COEFF'):
            late_coeff = float(line.split(':')[1])
        i += 1

    original_locations = locations[:, 1:]
    locations = original_locations / 1000
    demand = demand[:, 1:].squeeze()
    time_window = time_window[:, 1:]

    max_time = get_max_time(time_window)

    instance = VRPInstance(
        nb_customers        = dimension - 1, 
        locations           = locations, 
        original_locations  = original_locations, 
        demand              = demand, 
        capacity            = capacity,
        time_window         = time_window,
        service_time        = service_time,
        max_time            = max_time,
        late_coeff          = late_coeff)
        
    return instance


def read_instance_sd(path):
    raise NotImplementedError # has to be updated for VRPTW
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    locations = original_locations / (original_locations[0, 0] * 2)
    demand = demand[:, 1:].squeeze()

    instance = VRPInstance(dimension - 1, locations, original_locations, demand, capacity)
    return instance


def read_instances_pkl(path, offset=0, num_samples=None):
    # has too be updated for VRPTW
    raise NotImplementedError
    instances = []

    with open(path, 'rb') as f:
        data = pickle.load(f)

    if num_samples is None:
        num_samples = len(data)

    for args in data[offset:offset + num_samples]:
        depot, loc, demand, capacity, *args = args
        loc.insert(0, depot)
        demand.insert(0, 0)

        locations = np.array(loc)
        demand = np.array(demand)

        instance = VRPInstance(len(loc) - 1, locations, locations, demand, capacity)
        instances.append(instance)

    return instances


def read_dataset(path, size, create_solution):
    # 1) Training set 
    # 1a) Reading instances from dir 
    inst_set = []
    if os.path.isdir(path):
        instances = [ins for ins in os.listdir(path) if os.path.isfile(os.path.join(path, ins))]
        #instances = [ins for ins in instances if os.path.splitext(ins)[1] in ['.vrp']]
        if len(instances) > 0:
            logging.info(f"Found {len(instances)} instances")
            data_format = get_format(os.path.join(path, instances[0]))
            if data_format == 'cordeau':
                os.mkdir(os.path.join(path, '_vrplib'))
                for instance in instances:
                    out_path = convert_cordeau_to_vrplib(instance)
                    logging.info(f'Converted {instance} to {out_path}')
                instances = [ins for ins in os.listdir(path) if os.path.isfile(os.path.join(path, ins))]

            instances = [ins for ins in instances if os.path.splitext(ins)[1] in ['.vrp']]

            if len(instances) > size:
                #select the first size instances
                instances = instances[:size]
                logging.info(f"Selected the first {size} instances")
            elif len(instances) < size:
                raise ValueError(f"There are {len(instances)} instances in folder but model was expecting size = {size}")
# 1b) Co    nverting instances to VRPInstance objects #convert to mdvrpinstance list
            logging.info("Converting instances from files to VRPInstance list and creating initial solutions if required...")
            for el in tqdm(instances):
                instance = read_instance_vrp(os.path.join(path, el))
                if create_solution:
                    instance.create_initial_solution()
                inst_set.append(instance)
            logging.info("...done")
        else: 
            raise ValueError(f"Empty folder provided")

            

    # 1c) Reading instances from pkl 
    elif os.path.splitext(path)[1] == '.pkl':
        logging.info("Converting instances from pkl file to VRPInstance list...")
        with open(path, "rb") as f:
            inst_set = pickle.load(f)
            assert isinstance(inst_set[0], VRPInstance)
        logging.info("...done")
        if create_solution:
            logging.info("Creating initial solutions...")
            for el in tqdm(inst_set):
                instance.create_initial_solution()
            logging.info("...done")

    return inst_set

def get_format(path):
    lines = []
    with open(path) as f:
        lines = [line.rstrip() for line in f]
    first_line = line[0].split()
    if len(first_line) == 4 and first_line[0] in ['0', '1', '2', '3', '4', '5', '6', '7',]:
        return 'cordeau'
    else:
        return 'vrplib'

def save_dataset_pkl(instances, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(instances, f)
    print(f"Saved dataset to {output_path}")

class PlotRoute:
    """
    A lightweight route: a sequence of client indices (no depots inside),
    plus the start/end depot indices. Created to be plottable by plot_solution of PyVRP.
    """
    def __init__(self, visits: Iterable[int], start_depot: int, end_depot: int):
        self._visits: List[int] = [int(v) for v in visits]   # clients only
        self._start = int(start_depot)
        self._end = int(end_depot)

        if len(self._visits) == 0:
            raise ValueError("PlotRoute requires at least one client for plot_solution().")

    # --- what plot_solution needs ---
    def __len__(self) -> int:
        return len(self._visits)

    def __iter__(self):
        return iter(self._visits)

    def __getitem__(self, i: int) -> int:
        return self._visits[i]

    def __array__(self, dtype=None):
        # lets NumPy treat `route` as an index array: x_coords[route]
        arr = np.asarray(self._visits, dtype=np.intp)
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def start_depot(self) -> int:
        return self._start

    def end_depot(self) -> int:
        return self._end


class PlotSolution:
    """
    Minimal "solution" wrapper exposing .routes() -> iterable[PlotRoute]. Created to be plottable by plot_solution of PyVRP.
    """
    def __init__(self, routes: Iterable[PlotRoute]):
        self._routes = list(routes)

    def routes(self) -> Iterable[PlotRoute]:
        return self._routes

def vrp_to_plot_solution(inst) -> PlotSolution:
    """
    Converts VRPInstance.solution (list of tours of [node, demand, nn_idx])
    into a PlotSolution that plot_solution() can draw.

    Keeps only complete tours that start and end at a depot and have >= 1 client.
    """
    assert inst.solution is not None, "Instance has no solution to plot."

    routes: List[PlotRoute] = []
    depot_set = set([0])

    for tour in inst.solution:
        # Skip depot placeholders or incomplete tours
        if len(tour) < 3:
            continue
        start, end = tour[0][0], tour[-1][0]
        #if start not in depot_set or end not in depot_set:
        #    # Incomplete: skip (or handle specially if you want to visualize partial routes)
        #    continue

        # Middle of the tour should be only clients; still filter out any accidental depots
        visits = [node for (node, _, _) in tour[1:-1] if node not in depot_set]
        if visits:
            routes.append(PlotRoute(visits=visits, start_depot=start, end_depot=end))

    return PlotSolution(routes)
