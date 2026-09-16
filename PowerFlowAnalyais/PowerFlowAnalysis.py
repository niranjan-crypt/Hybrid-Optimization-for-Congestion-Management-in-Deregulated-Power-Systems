import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.converter as pc
import pandapower.plotting as plot
import cmath
import math

net = pp.create_empty_network(name="Converted Network", f_hz=50.0)

BMva = 100.0

# [Bus, Type, Vsp, theta, PGi, QGi, PLi, QLi, Qmin, Qmax]
busdata = np.array([
    [1, 1, 1.06, 0, 145, 0, 0, 0, -500, 500],
    [2, 2, 1.00, 0, 80, 0, 20, 10, -300, 300],
    [3, 3, 1.00, 0, 0, 0, 45+1.288, 15, 0, 0],
    [4, 3, 1.00, 0, 0, 0, 40-2.578, 5, 0, 0],
    [5, 3, 1.00, 0, 0, 0, 52+2.1688-2.892, 10, 0, 0]
])

gen = busdata[1, 4] - 2.1688

# [from_bus, to_bus, r, x, b, tap]
linedata =np.array([
    [1, 2, 0.02, 0.05, 0.03, 1],
    [1, 3, 0.08, 0.20, 0.025, 1],
    [2, 3, 0.06, 0.15, 0.02, 1],
    [2, 4, 0.06, 0.15, 0.02, 1],
    [2, 5, 0.04, 0.10, 0.015, 1],
    [3, 4, 0.01, 0.02, 0.01, 1],
    [4, 5, 0.08, 0.20, 0.025, 1]
]) 
# np.array([
#     [1, 2, 0.0192, 0.0575, 0.0264, 1],
#     [1, 3, 0.0452, 0.1852, 0.0204, 1],
#     [2, 4, 0.0570, 0.1737, 0.0184, 1],
#     [3, 4, 0.0132, 0.0379, 0.0042, 1],
#     [2, 5, 0.0472, 0.1983, 0.0209, 1],
#     [2, 6, 0.0581, 0.1763, 0.0187, 1],
#     [4, 6, 0.0119, 0.0414, 0.0045, 1],
#     [5, 7, 0.0460, 0.1160, 0.0102, 1],
#     [6, 7, 0.0267, 0.0820, 0.0085, 1],
#     [6, 8, 0.0120, 0.0420, 0.0045, 1],
#     [6, 9, 0.0000, 0.2080, 0.0000, 1.0155],
#     [6, 10, 0.0000, 0.5560, 0.0000, 1],
#     [9, 11, 0.0000, 0.2080, 0.0000, 1],
#     [9, 10, 0.0000, 0.1100, 0.0000, 1],
#     [4, 12, 0.0000, 0.2560, 0.0000, 1.0129],
#     [12, 13, 0.0000, 0.1400, 0.0000, 1],
#     [12, 14, 0.1231, 0.2559, 0.0000, 1],
#     [12, 15, 0.0662, 0.1304, 0.0000, 1],
#     [12, 16, 0.0945, 0.1987, 0.0000, 1],
#     [14, 15, 0.2210, 0.1997, 0.0000, 1],
#     [16, 17, 0.0824, 0.1932, 0.0000, 1],
#     [15, 18, 0.1070, 0.2185, 0.0000, 1],
#     [18, 19, 0.0639, 0.1292, 0.0000, 1],
#     [19, 20, 0.0340, 0.0680, 0.0000, 1],
#     [10, 20, 0.0936, 0.2090, 0.0000, 1],
#     [10, 17, 0.0324, 0.0845, 0.0000, 1],
#     [10, 21, 0.0348, 0.0749, 0.0000, 1],
#     [10, 22, 0.0727, 0.1499, 0.0000, 1],
#     [21, 22, 0.0116, 0.0236, 0.0000, 1],
#     [15, 23, 0.1000, 0.2020, 0.0000, 1],
#     [22, 24, 0.1150, 0.1790, 0.0000, 1],
#     [23, 24, 0.1320, 0.2700, 0.0000, 1],
#     [24, 25, 0.1885, 0.3292, 0.0000, 1],
#     [25, 26, 0.2544, 0.3800, 0.0000, 1],
#     [25, 27, 0.1093, 0.2087, 0.0000, 1],
#     [28, 27, 0.0000, 0.3690, 0.0000, 0.9581],
#     [27, 29, 0.2198, 0.4153, 0.0000, 1],
#     [27, 30, 0.3202, 0.6027, 0.0000, 1],
#     [29, 30, 0.2399, 0.4533, 0.0000, 1],
#     [8, 28, 0.0636, 0.2000, 0.0214, 1],
#     [6, 28, 0.0169, 0.0599, 0.0065, 1]
# ])

all_buses = set()
for line in linedata:
    all_buses.add(int(line[0]))
    all_buses.add(int(line[1]))

for bus_id in sorted(all_buses):
    bus_match = [i for i, row in enumerate(busdata) if int(row[0]) == bus_id]
    
    if bus_match:
        i = bus_match[0]
        bus_type = int(busdata[i, 1])
        if bus_type == 1:
            pp_type = "slack"
        elif bus_type == 2:
            pp_type = "gen"
        else:
            pp_type = "load"
    else:
        pp_type = "load"
    
    pp.create_bus(net, vn_kv=100.0, min_vm_pu=0.9, max_vm_pu=1.1, 
                  name=f"Bus {bus_id}", index=bus_id-1)
# Add loads
for i in range(len(busdata)):
    bus_id = int(busdata[i, 0]) - 1  
    pl = busdata[i, 6]
    ql = busdata[i, 7]
    
    if pl != 0 or ql != 0:
        pp.create_load(net, bus=bus_id, p_mw=pl, q_mvar=ql, name=f"Load at Bus {bus_id+1}")

# Add generators
for i in range(len(busdata)):
    bus_id = int(busdata[i, 0]) - 1  
    bus_type = int(busdata[i, 1])
    v_sp = busdata[i, 2]
    pg = busdata[i, 4]
    qg = busdata[i, 5]
    qmin = busdata[i, 8]
    qmax = busdata[i, 9]
    
    if bus_type == 1:  
        pp.create_ext_grid(net, bus=bus_id, vm_pu=v_sp, va_degree=0.0, 
                         name=f"Slack at Bus {bus_id+1}")
    elif bus_type == 2:  
        pp.create_gen(net, bus=bus_id, vm_pu=v_sp, p_mw=pg, 
                     min_q_mvar=qmin, max_q_mvar=qmax, 
                     name=f"Generator at Bus {bus_id+1}")

for i in range(len(linedata)):
    from_bus = int(linedata[i, 0]) - 1  
    to_bus = int(linedata[i, 1]) - 1    
    r = linedata[i, 2]
    x = linedata[i, 3]
    b = linedata[i, 4]
    tap = linedata[i, 5]
    
    if tap == 1.0:  
        pp.create_line_from_parameters(net, from_bus=from_bus, to_bus=to_bus, 
                                     length_km=47.0, r_ohm_per_km=r, x_ohm_per_km=x, 
                                     c_nf_per_km=b*1e9, max_i_ka=1.0,
                                     index=i,  
                                     name=f"Line {from_bus+1}-{to_bus+1}",type="ol")
    else:  
        pp.create_transformer_from_parameters(net, hv_bus=from_bus, lv_bus=to_bus,
                                           sn_mva=100, vn_hv_kv=100.0, vn_lv_kv=10.0,
                                           vk_percent=x*100, vkr_percent=r*100,
                                           pfe_kw=0, i0_percent=0, 
                                           tap_side="hv", tap_pos=int((tap-1)*100),
                                           tap_step_percent=1.0,
                                           index=i,  
                                           name=f"Transformer {from_bus+1}-{to_bus+1}")
# Run Newton-Raphson power flow
try:
    pp.runpp(net, algorithm='nr', calculate_voltage_angles=True, init="dc", 
          max_iteration=200, tolerance_mva=1e-5)
    print("Power flow calculation successful")
except Exception as e:
    print(f"Power flow calculation failed: {e}")


print('#########################################################################################')
print('-----------------------------------------------------------------------------------------')
print('                              Newton Raphson Loadflow Analysis ')
print('-----------------------------------------------------------------------------------------')
print('| Bus |    V   |  Angle  |     Injection      |     Generation     |          Load      |')
print('| No  |   pu   |  Degree |    MW   |   MVar   |    MW   |  Mvar    |     MW     |  MVar | ')

bus_results = net.res_bus
for i, bus in enumerate(net.bus.index):
    vm_pu = net.res_bus.vm_pu[i]
    va_degree = net.res_bus.va_degree[i]
    
    p_injection = 0
    q_injection = 0
    p_gen = 0
    q_gen = 0
    p_load = 0
    q_load = 0
    
    for load_idx in net.load.index:
        if net.load.bus[load_idx] == bus:
            p_load += net.load.p_mw[load_idx]
            q_load += net.load.q_mvar[load_idx]

    for gen_idx in net.gen.index:
        if net.gen.bus[gen_idx] == bus:
            p_gen += net.res_gen.p_mw[gen_idx]
            q_gen += net.res_gen.q_mvar[gen_idx]
    
    for ext_grid_idx in net.ext_grid.index:
        if net.ext_grid.bus[ext_grid_idx] == bus:
            p_gen += net.res_ext_grid.p_mw[ext_grid_idx]
            q_gen += net.res_ext_grid.q_mvar[ext_grid_idx]

    p_injection = p_gen - p_load
    q_injection = q_gen - q_load
    
    print('-----------------------------------------------------------------------------------------')
    print(f'{bus+1:3g}  {vm_pu:8.4f}   {va_degree:8.4f}  {p_injection:8.2f}   {q_injection:8.2f}  '
          f'{p_gen:8.2f}   {q_gen:8.2f}  {p_load:8.2f}   {q_load:8.2f}')

total_p_injection = sum(net.res_ext_grid.p_mw) + sum(net.res_gen.p_mw) - sum(net.load.p_mw)
total_q_injection = sum(net.res_ext_grid.q_mvar) + sum(net.res_gen.q_mvar) - sum(net.load.q_mvar)
total_p_gen = sum(net.res_ext_grid.p_mw) + sum(net.res_gen.p_mw)
total_q_gen = sum(net.res_ext_grid.q_mvar) + sum(net.res_gen.q_mvar)
total_p_load = sum(net.load.p_mw)
total_q_load = sum(net.load.q_mvar)

print('-----------------------------------------------------------------------------------------')
print(f' Total                  {total_p_injection:8.3f}   {total_q_injection:8.3f}  '
      f'{total_p_gen:8.3f}   {total_q_gen:8.3f}  {total_p_load:8.3f}   {total_q_load:8.3f}')
print('-----------------------------------------------------------------------------------------')
print('#########################################################################################')

print('-------------------------------------------------------------------------------------')
print('                              Line Flow and Losses ')
print('-------------------------------------------------------------------------------------')
print('|From|To |    P    |    Q     | From| To |    P     |   Q     |      Line Loss      |')
print('|Bus |Bus|   MW    |   MVar   | Bus | Bus|    MW    |  MVar   |     MW   |    MVar  |')

line_results = pd.DataFrame()
if len(net.res_line) > 0:
    line_results = pd.concat([
        net.res_line[['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'pl_mw', 'ql_mvar']],
        pd.DataFrame({
            'from_bus': net.line['from_bus'],
            'to_bus': net.line['to_bus']
        })
    ], axis=1)

transformer_results = pd.DataFrame()
if len(net.res_trafo) > 0:
    transformer_results = pd.concat([
        net.res_trafo[['p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar', 'pl_mw', 'ql_mvar']],
        pd.DataFrame({
            'from_bus': net.trafo['hv_bus'],
            'to_bus': net.trafo['lv_bus']
        })
    ], axis=1)
    transformer_results = transformer_results.rename(columns={
        'p_hv_mw': 'p_from_mw', 'q_hv_mvar': 'q_from_mvar',
        'p_lv_mw': 'p_to_mw', 'q_lv_mvar': 'q_to_mvar'
    })


all_results = pd.concat([line_results, transformer_results], ignore_index=True)

for idx, row in all_results.iterrows():
    from_bus = int(row['from_bus'] + 1)  
    to_bus = int(row['to_bus'] + 1)      
    
    p_from = row['p_from_mw']
    q_from = row['q_from_mvar']
    p_to = row['p_to_mw']
    q_to = row['q_to_mvar']
    pl = row['pl_mw']
    ql = row['ql_mvar']
    
    print('-------------------------------------------------------------------------------------')
    print(f'{from_bus:4g}{to_bus:4g}  {p_from:8.2f}   {q_from:8.2f}   '
          f'{to_bus:4g}{from_bus:4g}   {p_to:8.2f}   {q_to:8.2f}  '
          f'{pl:8.2f}   {ql:8.2f}')

# Calculate total losses
total_pl = all_results['pl_mw'].sum()
total_ql = all_results['ql_mvar'].sum()

print('-------------------------------------------------------------------------------------')
print(f'   Total Loss                                                 {total_pl:8.3f}   {total_ql:8.3f}')
print('-------------------------------------------------------------------------------------')
print('#####################################################################################')

actual_line_flows = []
for idx, row in all_results.iterrows():
    actual_flow = max(abs(row['p_from_mw']), abs(row['p_to_mw']))
    actual_line_flows.append(actual_flow)

actual_line_flows = np.array(actual_line_flows)
print("Actual Line Flows:")
print(actual_line_flows)

# Line flow limits
limit = np.array([100.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
print("Line Flow Limits:")
print(limit)


d = limit - actual_line_flows[:len(limit)]
print("Difference between limit and actual flow (d):")
print(d)

# Calculate the cost
cost = 200 * gen + 100 * (gen ** 2)
print(f"Generation Cost: {cost}")

