import numpy as np
import torch
import torch.optim as optim
import os
from search import destroy_instances
from copy import deepcopy
import logging
import datetime
from search_batch import lns_batch_search
import repair
import main
from vrp.data_utils import create_dataset, read_dataset
from search import LnsOperatorPair
from tqdm import tqdm, trange
import wandb

def get_datasets(config, rng):
    # first check if the datasets exist 
    if config.val_dataset is not None:
        assert os.path.exists(config.val_dataset)
    if config.train_dataset is not None:
        assert os.path.exists(config.train_dataset)

    # then either read them or create them
    batch_size = config.batch_size
    if config.train_dataset is None:
        logging.info("Generating training data...")
        # Create training and validation set. The initial solutions are created greedily
        training_set = create_dataset(
            size            = batch_size * config.nb_batches_training_set, 
            config          = config,
            create_solution = True, 
            use_cost_memory = False,
            rng             = rng)
    else: 
        logging.info(f"Reading training data from {config.train_dataset}...")
        training_set = read_dataset(
            path            = config.train_dataset, 
            size            = batch_size*config.nb_batches_training_set, 
            create_solution = True, 
            )

    if config.train_dataset is None:
        assert config.valid_size % config.lns_batch_size == 0, 'Validation size is not a multiple of lns_batch_size'
        logging.info("Generating validation data...")
        validation_instances = create_dataset(
            size            = config.valid_size, 
            config          = config, 
            create_solution = True,
            rng             = rng)
    else:
        logging.info(f"Reading validation data from {config.val_dataset}")
        assert os.path.isdir(config.val_dataset)
        validation_instances = read_dataset(
            path            = config.val_dataset, 
            size            = config.valid_size, 
            create_solution = True,
            )

    assert len(validation_instances) % config.lns_batch_size == 0

    return training_set, validation_instances

def train_nlns(actor, critic, run_id, config):
    #set up seed for reproducibility
    if config.seed is not None:
        rng = np.random.default_rng(config.seed)
    else:
        rng = np.random.default_rng()

    training_set, validation_instances = get_datasets(config, rng)

    actor_optim = optim.Adam(actor.parameters(), lr=config.actor_lr)
    actor.train()
    critic_optim = optim.Adam(critic.parameters(), lr=config.critic_lr)
    critic.train()

    losses_actor, rewards, diversity_values, losses_critic = [], [], [], []
    incumbent_costs = np.inf
    start_time = datetime.datetime.now()
    log_f = config.log_f

    if config.wandb:
        # wandb logging
        wandb.init(
            project="vrptw-nlns",
            id=str(run_id),
            tags=["training"],
            config=config,
        )
        wandb.define_metric('batch_idx')
        wandb.define_metric('train/*', step_metric='batch_idx')

        wandb.watch(actor, log='all', log_freq=log_f)

    logging.info("Starting training...")
    batch_size = config.batch_size
    for batch_idx in trange(1, config.nb_train_batches + 1):
        # Get a batch of training instances from the training set. Training instances are generated in advance, because
        # generating them is expensive.
        training_set_batch_idx = batch_idx % config.nb_batches_training_set
        tr_instances = [deepcopy(instance) for instance in
                        training_set[training_set_batch_idx * batch_size: (training_set_batch_idx + 1) * batch_size]]

        # Destroy and repair the set of instances
        destroy_instances(rng, tr_instances, config.lns_destruction, config.lns_destruction_p)
        costs_destroyed = [instance.get_costs_incomplete(config.round_distances) for instance in tr_instances]
        tour_indices, tour_logp, critic_est = repair.repair(tr_instances, actor, config, critic, rng)
        costs_repaired = [instance.get_costs(config.round_distances) for instance in tr_instances]
        late_mins = [instance.get_sum_late_mins() for instance in tr_instances]
        distance = [instance.get_total_distance() for instance in tr_instances]

        #getting percentages of cost composed by dist or delay
        perc_dist = sum(distance)/sum(costs_repaired)
        perc_delay = 1 - perc_dist

        # Reward/Advantage computation
        if config.scale_rewards:
            reward = (np.array(costs_repaired) - np.array(costs_destroyed))/ np.array(costs_destroyed)
        else:
            reward = np.array(costs_repaired) - np.array(costs_destroyed)

        reward = torch.from_numpy(reward).float().to(config.device)
        advantage = reward - critic_est

        # Actor loss computation and backpropagation
        actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
        actor_optim.zero_grad()
        actor_loss.backward()
        total = 0.0
        
        #DEBUG
        n_params = 0
        n_none = 0
        sum_abs = 0.0
        for p in actor.parameters():
            n_params += 1
            if p.grad is None:
                n_none += 1
            else:
                sum_abs += p.grad.detach().abs().sum().item()
        print(f"DEBUG: grads: none/total =", n_none, "/", n_params, " sum|grad|=", sum_abs)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        actor_optim.step()

        # Critic loss computation and backpropagation
        critic_loss = torch.mean(advantage ** 2)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        critic_optim.step()

        rewards.append(torch.mean(reward.detach()).item())
        losses_actor.append(torch.mean(actor_loss.detach()).item())
        losses_critic.append(torch.mean(critic_loss.detach()).item())

        # Replace the solution of the training set instances with the new created solutions
        for i in range(batch_size):
            training_set[training_set_batch_idx * batch_size + i] = tr_instances[i]

        if batch_idx % log_f == 0 and batch_idx > 0:
            mean_loss = np.mean(losses_actor[-log_f:])  #avg actor loss over the last log_f batches
            mean_critic_loss = np.mean(losses_critic[-log_f:]) #avg critic loss over the last log_f batches
            mean_reward = np.mean(rewards[-log_f:]) #avg reward of last log_f batches
            # cost of this batch (multiple of log_f)
            train_cost_batch = np.mean(costs_repaired) # mean repair cost OF THE CURRENT BATCH
            train_cost_batch_destroyed = np.mean(costs_destroyed) 


            # validation costs
            #val_cost_snapshot = lns_validation_search(validation_instances, actor, config, rng) # mean lns cost over validation_instances

            if config.wandb:
                wandb.log({
                    'batch_idx': int(batch_idx), 
                    'train/reward': float(mean_reward), 
                    'train/actor_loss': float(mean_loss), 
                    'train/critic_loss': float(mean_critic_loss),
                    'train/train_cost_batch': float(train_cost_batch),
                    'train/train_cost_batch_destroyed': float(train_cost_batch_destroyed),
                    'train/min_costs_destroyed_batch': float(min(costs_destroyed)),
                    'train/max_costs_destroyed_batch': float(max(costs_destroyed)),
                    #'train/val_cost_snapshot': float(val_cost_snapshot),
                    'train/sum_late_mins': int(sum(late_mins)),
                    'train/sum_distance': int(sum(distance)),
                    'train/perc_dist': perc_dist,
                    'train/perc_delay': perc_delay,
                })

        # Log performance every 250 batches
        if batch_idx % 250 == 0 and batch_idx > 0:
            mean_loss = np.mean(losses_actor[-250:])
            mean_critic_loss = np.mean(losses_critic[-250:])
            mean_reward = np.mean(rewards[-250:])
            logging.info(
                f'Batch {batch_idx}/{config.nb_train_batches}, repair costs (reward): {mean_reward:2.3f}, loss: {mean_loss:2.6f}'
                f', critic_loss: {mean_critic_loss:2.6f}')

        # Evaluate and save model every 200 batches
        if batch_idx % 200 == 0 or batch_idx == config.nb_train_batches:
            mean_costs = lns_validation_search(validation_instances, actor, config, rng)
            model_data = {
                'parameters': actor.state_dict(),
                'model_name': "VrpActorModel",
                'destroy_operation': config.lns_destruction,
                'p_destruction': config.lns_destruction_p,
                'code_version': main.VERSION
            }

            if config.split_delivery:
                problem_type = "SD"
            else:
                problem_type = "C"
            torch.save(model_data, os.path.join(config.output_path, "models",
                                                "model_{0}_{1}_{2}_{3}_{4}.pt".format(problem_type,
                                                                                      config.instance_blueprint,
                                                                                      config.lns_destruction,
                                                                                      config.lns_destruction_p,
                                                                                      run_id)))
            if mean_costs < incumbent_costs:
                incumbent_costs = mean_costs
                incumbent_model_path = os.path.join(config.output_path, "models",
                                                    "model_incumbent_{0}_{1}_{2}_{3}_{4}.pt".format(problem_type,
                                                                                                    config.instance_blueprint,
                                                                                                    config.lns_destruction,
                                                                                                    config.lns_destruction_p,
                                                                                                    run_id))
                torch.save(model_data, incumbent_model_path)

            runtime = (datetime.datetime.now() - start_time)
            logging.info(
                f"Validation (Batch {batch_idx}) Costs: {mean_costs:.3f} ({incumbent_costs:.3f}) Runtime: {runtime}")
    if config.wandb:
        wandb.finish()
    return incumbent_model_path


def lns_validation_search(validation_instances, actor, config, rng):
    validation_instances_copies = [deepcopy(instance) for instance in validation_instances]
    actor.eval()
    operation = LnsOperatorPair(actor, config.lns_destruction, config.lns_destruction_p)
    costs, _ = lns_batch_search(validation_instances_copies, config.lns_max_iterations,
                                config.lns_timelimit_validation, [operation], config, rng)
    actor.train()
    return np.mean(costs)
