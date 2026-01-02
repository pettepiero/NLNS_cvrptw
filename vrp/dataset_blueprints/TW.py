from vrp.data_utils import InstanceBlueprint

dataset = {}
dataset['10'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=1)  
dataset['20'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=0.1)  
dataset['30'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=0.001)  
dataset['40'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=0)  
dataset['50'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=0)  
