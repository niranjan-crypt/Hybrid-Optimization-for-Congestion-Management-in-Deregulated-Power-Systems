import pypsa
import pandas as pd

# Create a new network
network = pypsa.Network()

# IEEE 30-bus system data
bus_data = [
    (1, 'Slack', 1.06, 0), (2, 'PQ', 1.043, None), (3, 'PQ', 1.021, None), (4, 'PQ', 1.012, None),
    (5, 'PQ', 1.01, None), (6, 'PQ', 1.01, None), (7, 'PQ', 1.002, None), (8, 'PQ', 1.01, None),
    (9, 'PQ', 1.051, None), (10, 'PQ', 1.045, None), (11, 'PQ', 1.082, None), (12, 'PQ', 1.057, None),
    (13, 'PQ', 1.05, None), (14, 'PQ', 1.036, None), (15, 'PQ', 1.024, None), (16, 'PQ', 1.022, None),
    (17, 'PQ', 1.021, None), (18, 'PQ', 1.019, None), (19, 'PQ', 1.015, None), (20, 'PQ', 1.01, None),
    (21, 'PQ', 1.01, None), (22, 'PQ', 1.01, None), (23, 'PQ', 1.01, None), (24, 'PQ', 1.01, None),
    (25, 'PQ', 1.01, None), (26, 'PQ', 1.01, None), (27, 'PQ', 1.01, None), (28, 'PQ', 1.01, None),
    (29, 'PQ', 1.01, None), (30, 'PQ', 1.01, None)
]

for bus_id, bus_type, v_nom, angle in bus_data:
    network.add("Bus", str(bus_id), v_nom=v_nom)

# Add generators
# Ensure that Bus 1 generator is set as the Slack generator
generators = [
    ("Gen1", "1", 50, 10, 1.06, "Slack"), ("Gen2", "2", 63.75, 20, 1.043, "PV"),
    ("Gen3", "5", 82.875, 30, 1.01, "PV"), ("Gen4", "8", 93.75, 40, 1.01, "PV"),
    ("Gen5", "11", 157.5, 50, 1.082, "PV"), ("Gen6", "13", 176.625, 60, 1.05, "PV")
]

for name, bus, p_set, min_p, v_set, control in generators:
    network.add("Generator", name, bus=str(bus), p_set=p_set, control=control, v_set=v_set)

# Add loads
loads = [
    ("Load1", "2", 21.7), ("Load2", "3", 94.2), ("Load3", "4", 47.8),
    ("Load4", "5", 7.6), ("Load5", "7", 22.8), ("Load6", "8", 30.0),
    ("Load7", "10", 5.8), ("Load8", "12", 11.2), ("Load9", "14", 6.2),
    ("Load10", "15", 8.2), ("Load11", "16", 3.5), ("Load12", "17", 9.0),
    ("Load13", "18", 3.2), ("Load14", "19", 9.5), ("Load15", "20", 2.2),
    ("Load16", "21", 17.5), ("Load17", "23", 3.2), ("Load18", "24", 8.7),
    ("Load19", "26", 3.5), ("Load20", "29", 2.4), ("Load21", "30", 10.6)
]

for name, bus, p_set in loads:
    network.add("Load", name, bus=str(bus), p_set=p_set)

# Perform power flow analysis
network.pf()  # AC power flow

# Display results in tabular form
print("\nBus Voltage Results:")
if "v_mag_pu" in network.buses.columns:
    print(network.buses[['v_nom', 'v_mag_pu']].to_markdown())
else:
    print("Voltage magnitude data not available.")

print("\nLine Flow Results:")
if "s_pu" in network.lines.columns:
    print(network.lines[['s_nom', 's_pu']].to_markdown())
else:
    print("Line flow data not available.")