# Parts of this code are based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py

import torch
import numpy as np
from vrp import vrp_problem 
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)

def _actor_model_forward(actor, instances, static_input, dynamic_input, config, vehicle_capacity, rng):
    batch_size = static_input.shape[0]
    N = static_input.shape[1]
    tour_idx, tour_logp = [], []

    instance_repaired = np.zeros(batch_size)

    origin_idx = np.zeros((batch_size), dtype=int)
    last_dim = torch.zeros((batch_size, N, 2), dtype=static_input.dtype, device=static_input.device)
    repair_round = 0
    while not instance_repaired.all():
        # if origin_idx == 0 select the next tour end that serves as the origin at random
        for i, instance in enumerate(instances):
            if origin_idx[i] == 0 and not instance_repaired[i]:
                if rng is None:
                    origin_idx[i] = np.random.choice(instance.open_nn_input_idx, 1).item()
                else:
                    origin_idx[i] = rng.choice(instance.open_nn_input_idx, 1).item()
            if i != 0:
                last_dim[i] = instance.get_last_dim(static_input[i], origin_idx[i], print_debug=False)
            else:
                last_dim[i] = instance.get_last_dim(static_input[i], origin_idx[i], print_debug=True)
                print(f"RETURNED: last_dim[i]: {last_dim[i]}")

        # Rescale customer demand based on vehicle capacity
        dynamic_input_last_dim = torch.cat((dynamic_input, last_dim), dim=-1)
        print(f"\nDEBUG: for instance 0:")
        print(f"instances[0].solution:")
        for el in instances[0].solution:
            print(el)
        print(f"instances[0].schedule:")
        for el in instances[0].schedule:
            print(el)
        print(f"instances[0].nn_input_idx_to_tour:")
        for j, el in enumerate(instances[0].nn_input_idx_to_tour):
            print(el)
        print(f"origin_idx[0]: {origin_idx[0]}")
        print(f"origin_tour: {instances[0].nn_input_idx_to_tour[origin_idx[0]][0]}")
        print(f"origin_pos: {instances[0].nn_input_idx_to_tour[origin_idx[0]][1]}")
        origin_cust = instances[0].nn_input_idx_to_tour[origin_idx[0]][0][instances[0].nn_input_idx_to_tour[origin_idx[0]][1]][0]
        print(f"origin_cust: {origin_cust}")
        print(f"origin_cust locations: {instances[0].locations[origin_cust]}")
        print(f"origin_cust scaled time_window: {instances[0].time_window[origin_cust]/instances[0].max_time}")
        print(f"origin_cust time_window: {instances[0].time_window[origin_cust]}")
        print(f"origin_cust demand: {instances[0].demand[origin_cust]}")
         

        print(f"DEBUG: torch.cat(static_input, dynamic_input_last_dim, dim=-1)")
        for j, el in enumerate(torch.cat((static_input, dynamic_input_last_dim), dim=-1)[0]):
            if j != origin_idx[0]:
                print('  ',j, el)
            else:
                print('->', j, el)
        del origin_cust

        mask, invert_connection = vrp_problem.get_mask(origin_idx, static_input, dynamic_input_last_dim, instances, config, vehicle_capacity)
        mask = mask.to(config.device).float()
        # mask: (batch, N) float/bool-like
        #row_ok = (mask.sum(dim=1) > 0)
        #if not row_ok.all():
        #    bad = torch.where(~row_ok)[0].tolist()
        #    raise RuntimeError(f"All-zero mask rows at batch indices: {bad}")
        #feas = mask.sum(dim=1)              # (batch,)
        #print("mask feasible actions: min/mean/max =",
        #      feas.min().item(), feas.float().mean().item(), feas.max().item())
        #print("rows with only 1 action:", (feas == 1).sum().item(), "/", feas.numel())



        dynamic_input_float = dynamic_input_last_dim.float()
        dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0] / float(vehicle_capacity)

        origin_static_input = static_input[torch.arange(batch_size), origin_idx]
        origin_dynamic_input_float = dynamic_input_float[torch.arange(batch_size), origin_idx]

        # Forward pass. Returns a probability distribution over the point (tour end or depot) that origin should be connected to
        probs = actor.forward(static_input, dynamic_input_float, origin_static_input, origin_dynamic_input_float, mask)
        probs = F.softmax(probs + mask.log(), dim=1)  # Set prob of masked tour ends to zero

        if actor.training:
            m = torch.distributions.Categorical(probs)

            # Sometimes an issue with Categorical & sampling on GPU; See:
            # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
            ptr = m.sample()
            while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                ptr = m.sample()
            logp = m.log_prob(ptr)
        else:
            prob, ptr = torch.max(probs, 1)  # Greedy selection
            logp = prob.log()

        # Perform action  for all instances sequentially
        nn_input_updates = []
        ptr_np = ptr.cpu().numpy()
        for i, instance in enumerate(instances):
            idx_from = origin_idx[i].item()
            idx_to = ptr_np[i]
            if invert_connection[i]:
                idx_from, idx_to = idx_to, idx_from
            #log.info(f"\t For instance {i}/{len(instances)} sampled: idx_from={idx_from}, idx_to={idx_to}")
            #log.info(f"\t It means tour_from: {instance.nn_input_idx_to_tour[idx_from][0]} | tour_to: {instance.nn_input_idx_to_tour[idx_to][0]}")
            #log.info(f"\t Used mask for above sample: {mask}")
            #log.info(f"\t Number of possible actions: {[int(sum(l).item()) for l in mask]}")
            if idx_from == 0 and idx_to == 0:  # No need to update in this case
                continue
            if i == 0:
                print_debug = True 
            else:
                print_debug = False
            print_debug = True
            if print_debug:
                print(f"\nDEBUG: instance: {i}")
                print(f"DEBUG: instance.solution:")
                for k, el in enumerate(instance.solution):
                    print(k, el)
                print(f"DEBUG: instance.schedule:")
                for k, el in enumerate(instance.schedule):
                    print(k, el)
                print(f"DEBUG: instance.nn_input_idx_to_tour:")
                for k, el in enumerate(instance.nn_input_idx_to_tour):
                    if k in instance.open_nn_input_idx:
                        print(' ', k, el)
                    else:
                        print('X', k, el)
                print(f"DEBUG: idx_from: {idx_from} | {instance.nn_input_idx_to_tour[idx_from]}")
                print(f"DEBUG: idx_to: {idx_to} | {instance.nn_input_idx_to_tour[idx_to]}")
            nn_input_update, cur_nn_input_idx = instance.do_action(idx_from, idx_to, print_debug)  # Connect origin to select point
            for s in nn_input_update:
                s.insert(0, i)
                nn_input_updates.append(s)

            # Update origin
            if len(instance.open_nn_input_idx) == 0:
                instance_repaired[i] = 1
                origin_idx[i] = 0  # If instance is repaired set origin to 0
            else:
                origin_idx[i] = cur_nn_input_idx  # Otherwise, set to tour end of the connect tour
            print(f"\nDEBUG: after do_action:")
            for j, el in enumerate(instance.nn_input_idx_to_tour):
                print(j, el)

        # Update network input
        nn_input_update = np.array(nn_input_updates, dtype=np.long)
        nn_input_update = torch.from_numpy(nn_input_update).to(config.device).long()
        dynamic_input[nn_input_update[:, 0], nn_input_update[:, 1]] = nn_input_update[:, 2:]

        logp = logp * (1. - torch.from_numpy(instance_repaired).float().to(config.device))
        tour_logp.append(logp.unsqueeze(1))
        tour_idx.append(ptr.data.unsqueeze(1))
        repair_round += 1

    tour_idx = torch.cat(tour_idx, dim=1)
    tour_logp = torch.cat(tour_logp, dim=1)
    #print("DEBUG: tour_logp requires_grad:", tour_logp.requires_grad)
    #print("DEBUG: tour_logp grad_fn:", tour_logp.grad_fn)
    #print("DEBUG: tour_logp.isfinite:", torch.isfinite(tour_logp).all().item())
    #print("logp stats: min/mean/max =",
    #  logp.min().item(), logp.mean().item(), logp.max().item())

    return tour_idx, tour_logp


def _critic_model_forward(critic, static_input, dynamic_input, batch_capacity):
    dynamic_input_float = dynamic_input.float()

    dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0] / float(batch_capacity)

    return critic.forward(static_input, dynamic_input_float).view(-1)


def repair(instances, actor, config, critic=None, rng=None):
    nb_input_points = max([instance.get_max_nb_input_points() for instance in instances])  # Max. input points of batch
    batch_size = len(instances)

    # Create batch input
    static_input = np.zeros((batch_size, nb_input_points, 4))
    dynamic_input = np.zeros((batch_size, nb_input_points, 2), dtype='int')
    for i, instance in enumerate(instances):
        static_nn_input, dynamic_nn_input = instance.get_network_input(nb_input_points)
        static_input[i] = static_nn_input
        dynamic_input[i] = dynamic_nn_input

    static_input = torch.from_numpy(static_input).to(config.device).float()
    dynamic_input = torch.from_numpy(dynamic_input).to(config.device).long()

    vehicle_capacity = instances[0].capacity # Assumes that the vehicle capcity is identical for all instances of the batch

    cost_estimate = None
    if critic is not None:
        cost_estimate = _critic_model_forward(critic, static_input, dynamic_input, vehicle_capacity)

    tour_idx, tour_logp = _actor_model_forward(actor, instances, static_input, dynamic_input, config, vehicle_capacity, rng)

    return tour_idx, tour_logp, cost_estimate
