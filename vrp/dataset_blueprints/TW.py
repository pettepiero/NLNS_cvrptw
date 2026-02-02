from vrp.data_utils import InstanceBlueprint

dataset = {}
# 10 customers
dataset['10'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=1)  
dataset['20'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=0.1)  
dataset['30'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=0.001)  
dataset['40'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=0)  
# test on max TW
dataset['45'] = InstanceBlueprint(nb_customers=10, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, test_tw=True, late_coeff=0)  



# 15 customers
dataset['240'] = InstanceBlueprint(nb_customers=15, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, late_coeff=0)  
dataset['245'] = InstanceBlueprint(nb_customers=15, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000, service_time=100, max_time=10000, test_tw=True, late_coeff=0)  


# 20 customers
dataset['50'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=0)  
        # test TW instance
dataset['55'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, test_tw=True, late_coeff=0)  

dataset['60'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=10)  
dataset['70'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=1)  
dataset['80'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=0.1)  
dataset['90'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=0.05)
dataset['100'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=0.2)
dataset['110'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=2)
dataset['120'] = InstanceBlueprint(nb_customers=20, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000, service_time=100, max_time=10000, late_coeff=0.5)

# 25 customers
dataset['approxsolomon25'] = InstanceBlueprint(nb_customers=25, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=30, capacity=200, grid_size=100, service_time=10, max_time=230, late_coeff=0, fixed_tw=100)  
