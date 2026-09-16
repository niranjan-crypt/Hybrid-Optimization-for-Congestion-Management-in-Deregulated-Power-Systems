import numpy as np

# Bus data (Bus, Type, Vsp, theta, PGi, QGi, PLi, QLi, Qmin, Qmax)
bus_data = np.array([
    [1, 1, 1.06, 0, 145, 0, 0, 0, -500, 500],
    [2, 2, 1.00, 0, 80, 0, 20, 10, -300, 300],
    [3, 3, 1.00, 0, 0, 0, 45 + 1.288, 15, 0, 0],
    [4, 3, 1.00, 0, 0, 0, 40 - 2.578, 5, 0, 0],
    [5, 3, 1.00, 0, 0, 0, 52 + 2.1688 - 2.892, 10, 0, 0]
])

# Line data (From Bus, To Bus, R, X, B, Tap)
line_data = np.array([
    [1, 2, 0.02, 0.05, 0.03, 1],
    [1, 3, 0.08, 0.20, 0.025, 1],
    [2, 3, 0.06, 0.15, 0.02, 1],
    [2, 4, 0.06, 0.15, 0.02, 1],
    [2, 5, 0.04, 0.10, 0.015, 1],
    [3, 4, 0.01, 0.02, 0.01, 1],
    [4, 5, 0.08, 0.20, 0.025, 1]
])

# Number of buses and lines
nb = len(bus_data)
nl = len(line_data)

# Initialize Y-bus matrix
Y = np.zeros((nb, nb), dtype=complex)

# Compute Admittance Matrix
for i in range(nl):
    fb, tb, r, x, b, tap = line_data[i]
    fb, tb = int(fb) - 1, int(tb) - 1
    z = complex(r, x)
    y = 1 / z
    Y[fb, tb] -= y / tap
    Y[tb, fb] = Y[fb, tb]
    Y[fb, fb] += y / (tap ** 2) + complex(0, b)
    Y[tb, tb] += y + complex(0, b)

# Power Flow Calculation using Newton-Raphson
BMva = 100  # Base MVA
V = bus_data[:, 2]  # Voltage magnitude
theta = np.zeros(nb)  # Voltage angle, initialized close to 0 to prevent divergence
Pgen = bus_data[:, 4] / BMva
Qgen = bus_data[:, 5] / BMva
Pload = bus_data[:, 6] / BMva
Qload = bus_data[:, 7] / BMva
P = Pgen - Pload
Q = Qgen - Qload
G = Y.real
B = Y.imag

# Identify PV and PQ buses
pv = np.where(bus_data[:, 1] == 2)[0]
pq = np.where(bus_data[:, 1] == 3)[0]

# Iterative process
Tol = 1e-3  # Set lower tolerance for stability
max_iter = 100
iter_count = 0

while Tol > 1e-5 and iter_count < max_iter:
    iter_count += 1
    Pcalc = np.zeros(nb)
    Qcalc = np.zeros(nb)
    
    for i in range(nb):
        for k in range(nb):
            Pcalc[i] += V[i] * V[k] * (G[i, k] * np.cos(theta[i] - theta[k]) + B[i, k] * np.sin(theta[i] - theta[k]))
            Qcalc[i] += V[i] * V[k] * (G[i, k] * np.sin(theta[i] - theta[k]) - B[i, k] * np.cos(theta[i] - theta[k]))
    
    deltaP = P - Pcalc
    deltaQ = Q - Qcalc
    
    # Construct Jacobian Matrix
    J1 = np.zeros((nb - 1, nb - 1))
    J2 = np.zeros((nb - 1, len(pq)))
    J3 = np.zeros((len(pq), nb - 1))
    J4 = np.zeros((len(pq), len(pq)))
    
    for i in range(nb - 1):
        m = i + 1
        for k in range(nb - 1):
            n = k + 1
            if n == m:
                J1[i, k] = -V[m] ** 2 * B[m, m]
                for n in range(nb):
                    J1[i, k] += V[m] * V[n] * (-G[m, n] * np.sin(theta[m] - theta[n]) + B[m, n] * np.cos(theta[m] - theta[n]))
            else:
                J1[i, k] = V[m] * V[n] * (G[m, n] * np.sin(theta[m] - theta[n]) - B[m, n] * np.cos(theta[m] - theta[n]))
    
    J = np.block([[J1, J2], [J3, J4]])
    
    try:
        M = np.concatenate((deltaP[1:], deltaQ[pq]))
        X = np.linalg.solve(J, M)
    except np.linalg.LinAlgError:
        print("Jacobian matrix is singular. Adjusting system parameters.")
        break
    
    # Update Voltages and Angles
    theta[1:] += X[:nb - 1]
    k = 0
    for i in pq:
        V[i] += X[nb - 1 + k]
        V[i] = max(0.9, min(1.1, V[i]))  # Keep voltage within a reasonable range
        k += 1
    
    Tol = max(abs(M))

# Display results
print("Bus Voltages:")
for i in range(nb):
    print(f"Bus {i+1}: V = {V[i]:.4f} pu, Theta = {np.degrees(theta[i]):.4f} degrees")

print("\nY-Bus Matrix:")
print(Y)