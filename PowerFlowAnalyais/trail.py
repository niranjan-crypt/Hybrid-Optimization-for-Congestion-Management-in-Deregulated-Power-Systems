import pypsa  # Import the PyPSA library for power system analysis [1, 2, 3]

# Define the power system network
network = pypsa.Network()  

# Add buses (nodes) to the network
network.add("bus", "bus1",  bus_type="slack", v_nom=1.05)  # Slack bus
network.add("bus", "bus2", bus_type="pv",v_nom=1.0) 
network.add("bus", "bus3", bus_type="pq",v_nom=1.0) 
network.add("bus", "bus4", bus_type="pq",v_nom=1.0) 
network.add("bus", "bus5", bus_type="pq",v_nom=1.0) 

# Add lines (branches) between buses
network.add("line", "line1",  from_bus="bus1", to_bus="bus2", r=0.02, x=0.06) 
network.add("line", "line2", from_bus="bus1", to_bus="bus3", r=0.08, x=0.24)
network.add("line", "line3", from_bus="bus2", to_bus="bus3", r=0.06, x=0.18)
network.add("line", "line4", from_bus="bus2", to_bus="bus4", r=0.06, x=0.18)
network.add("line", "line5", from_bus="bus2", to_bus="bus5", r=0.04, x=0.12)
network.add("line", "line6", from_bus="bus3", to_bus="bus4", r=0.01, x=0.03)
network.add("line", "line7", from_bus="bus4", to_bus="bus5", r=0.08, x=0.24)


# Add loads at specific buses
network.add("load", "load1", bus="bus2", p=29, q=0)
network.add("load", "load2", bus="bus3", p=84, q=0) 
network.add("load", "load3", bus="bus4", p=45, q=0)
network.add("load", "load4", bus="bus5", p=45, q=0) 


# Solve the load flow analysis 
network.solve_powerflow()  

# Access results
print("Voltage at bus 2:", network.buses_t.v_mag_pu["bus2"])  
print("Power flow on line 1:", network.lines_t.p_from["line1"])




