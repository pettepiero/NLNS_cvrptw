import os

def convert_to_vrp(input_file_path):
    # Prepare storage for data
    data = {
        'name': '',
        'capacity': '',
        'node_coord': [],
        'demand': [],
        'time_window': [],
        'service_time': '0'
    }
    
    current_section = None
    
    # Read and parse the input file
    with open(input_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line == 'EOF':
                continue
            
            # Parse Headers
            if line.startswith('name:'):
                data['name'] = line.split(':')[1].strip()
            elif line.startswith('capacity:'):
                data['capacity'] = line.split(':')[1].strip()
            
            # Detect Section Changes
            elif line == 'node_coord':
                current_section = 'node_coord'
            elif line == 'demand':
                current_section = 'demand'
            elif line == 'time_window':
                current_section = 'time_window'
            elif line == 'service_time':
                current_section = 'service_time'
            
            # Parse Section Data
            else:
                if current_section == 'node_coord':
                    data['node_coord'].append(line)
                elif current_section == 'demand':
                    data['demand'].append(line)
                elif current_section == 'time_window':
                    data['time_window'].append(line)
                elif current_section == 'service_time':
                    # Extract the service time from the first non-depot node (index 2)
                    parts = line.split()
                    if parts[0] == '2': 
                        data['service_time'] = parts[1]

    # Calculate Dimension
    dimension = len(data['node_coord'])
    
    # Construct output filename
    base_name = os.path.splitext(input_file_path)[0]
    output_file_path = f"{base_name}.vrp"
    
    # Write the new format
    with open(output_file_path, 'w') as f:
        f.write(f"NAME : {data['name']}\n")
        f.write(f"TYPE : CVRPTW\n")
        f.write(f"DIMENSION : {dimension}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {data['capacity']}\n")
        f.write(f"SERVICE_TIME : {data['service_time']}\n")
        f.write(f"LATE_COEFF : 0\n")
        
        f.write("NODE_COORD_SECTION\n")
        for line in data['node_coord']:
            f.write(f"{line}\n")
            
        f.write("DEMAND_SECTION\n")
        for line in data['demand']:
            f.write(f"{line}\n")
            
        f.write("TIME_WINDOW_SECTION\n")
        for line in data['time_window']:
            f.write(f"{line}\n")
            
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")

    print(f"Conversion complete: {output_file_path}")

# Usage
# convert_to_vrp('R101.txt')
