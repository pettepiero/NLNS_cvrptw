import logging
import numpy as np
import os
import time
import torch
import search_single
from vrp.data_utils import read_instances_pkl
import glob
import search_batch
from actor import VrpActorModel
from tqdm import tqdm

class LnsOperatorPair:

    def __init__(self, model, destroy_procedure, p_destruction):
        self.model = model
        self.destroy_procedure = destroy_procedure
        self.p_destruction = p_destruction


def destroy_instances(rng, instances, destroy_procedure=None, destruction_p=None):
    for j, instance in enumerate(instances):
        #print(f"DEBUG: destroying instance {j}")
        if destroy_procedure == "R":
            instance.destroy_random(destruction_p, rng=rng)
        elif destroy_procedure == "P":
            instance.destroy_point_based(destruction_p, rng=rng)
        elif destroy_procedure == "T":
            if j == 0:
                print_debug=True
            else:
                print_debug=False
            instance.destroy_tour_based(destruction_p, rng=rng, print_debug=print_debug)


def load_operator_pairs(path, config):
    if path.endswith('.pt'):
        model_paths = [path]
    else:
        model_paths = glob.glob(os.path.join(path, '*.pt'))

    if not model_paths:
        raise Exception(f"No operators found in {path}")

    lns_operator_pairs = []
    for model_path in model_paths:
        model_data = torch.load(model_path, config.device)

        actor = VrpActorModel(config.device, hidden_size=config.pointer_hidden_size).to(
            config.device)
        actor.load_state_dict(model_data['parameters'])
        actor.eval()

        operator_pair = LnsOperatorPair(actor, model_data['destroy_operation'], model_data['p_destruction'])
        lns_operator_pairs.append(operator_pair)
    return lns_operator_pairs


def evaluate_batch_search(config, model_path, rng):
    assert model_path is not None, 'No model path given'

    logging.info('### Batch Search ###')
    logging.info('Starting search')
    start_time = time.time()

    results = search_batch.lns_batch_search_mp(config, model_path, rng)

    runtime = (time.time() - start_time)
    instance_id, costs, iterations = [], [], []
    for r in results:
        instance_id.extend(list(range(len(r[1]) * r[0], len(r[1]) * (r[0] + 1))))
        costs.extend(r[1])
        iterations.append(r[2])

    path = os.path.join(config.output_path, "search", "nlns_batch_search_results.txt")
    np.savetxt(path, np.column_stack((instance_id, costs)), delimiter=',', fmt=['%i', '%f'])
    print(f"Saved results of batch search in {path}")
    #path = os.path.join(config.output_path, "search", 'results.txt')
    #np.savetxt(path, np.column_stack((instance_id, costs)), delimiter=',', fmt=['%i', '%f'])
    logging.info(
        f"Test set costs: {np.mean(costs):.3f} Total Runtime (s): {runtime:.1f} Iterations: {np.mean(iterations):.1f}")


def evaluate_single_search(config, model_path, instance_path):
    assert model_path is not None, 'No model path given'
    assert instance_path is not None, 'No instance path given'

    instance_names, instance_ids, costs, durations, distances, sums_late_mins = [], [], [], [], [], []
    logging.info("### Single instance search ###")

    if instance_path.endswith(".vrp") or instance_path.endswith(".sd"):
        logging.info("Starting solving a single instance")
        instance_files_path = [instance_path]
    elif instance_path.endswith(".pkl"):
        instance_files_path = [instance_path] * len(read_instances_pkl(instance_path))
        logging.info("Starting solving a .pkl instance set")
    elif os.path.isdir(instance_path):
        instance_files_path = [os.path.join(instance_path, f) for f in os.listdir(instance_path)]
        logging.info("Starting solving all instances in directory")
    else:
        raise Exception("Unknown instance file format.")

    for i, instance_path in enumerate(tqdm(instance_files_path)):
        if instance_path.endswith(".pkl") or instance_path.endswith(".vrp") or instance_path.endswith(".sd"):
            for _ in range(config.nb_runs):
                cost, duration, distance, sum_late_mins = search_single.lns_single_search_mp(instance_path, config.lns_timelimit, config,
                                                                    model_path, i, plot_sol = config.plot_sol)
                instance_names.append(instance_path)
                instance_ids.append(i)
                costs.append(cost)
                distances.append(distance)
                durations.append(duration)
                sums_late_mins.append(sum_late_mins)

    output_path_with_times = os.path.join(config.output_path, "search", 'nlns_batch_search_results_with_times.txt')
    output_path_distances = os.path.join(config.output_path, "search", 'nlns_batch_search_distances.txt')
    output_path_delay = os.path.join(config.output_path, "search", 'nlns_batch_search_delay.txt')
    output_path = os.path.join(config.output_path, "search", 'nlns_batch_search_results.txt')
    results_with_times = np.array(list(zip(instance_names, costs, durations)))
    results = np.array(list(zip(instance_ids, costs)))
    distances_results = np.array(list(zip(instance_ids, distances)))
    delay_results = np.array(list(zip(instance_ids, sums_late_mins)))
    np.savetxt(output_path_with_times, results_with_times, delimiter=',', fmt=['%s', '%s', '%s'], header="name, cost, runtime")
    np.savetxt(output_path_distances, distances_results, delimiter=',')
    np.savetxt(output_path_delay, delay_results, delimiter=',')
    np.savetxt(output_path, results, delimiter=',',)
    print(f"Saved results of single search in {output_path}")
    print(f"Saved distances of single search in {output_path_distances}")
    print(f"Saved delay of single search in {output_path_delay}")
    print(f"config.output_path = {config.output_path}")

    logging.info(
        f"NLNS single search evaluation results: Total Nb. Runs: {len(costs)}, "
        f"Mean Costs: {np.mean(costs):.3f} Mean Runtime (s): {np.mean(durations):.1f}"
        f"Mean Distance: {np.mean(distances):.3f} Mean Delay: {np.mean(sums_late_mins):.1f}")
