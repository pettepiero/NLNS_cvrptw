from vrp.data_utils import InstanceBlueprint

dataset = {}
dataset['1'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=2, capacity=4, grid_size=1000, service_time=10, late_coeff=1, max_time=10000)  
dataset['2'] = InstanceBlueprint(nb_customers=100, depot_position='R', customer_position='R', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=10, late_coeff=1, max_time=10000)  
dataset['3'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=10, late_coeff=1, max_time=10000)  
dataset['4'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=10, late_coeff=10, max_time=10000)  
