import numpy as np
import random
import matplotlib.pyplot as plt
def Optimizer(SensitivityMatrix, BusInjection, LineLimits, iterations=100):
    
    NoOfBuses = SensitivityMatrix.shape[1]
    BestSolution = BusInjection.copy()
    BestFitness = np.inf
    FitnessHistory = []
     
    PowerFlowInitial = np.dot(SensitivityMatrix,BusInjection)
    TotalPowerFlowInitial = np.sum(abs(PowerFlowInitial))
    
    for iterations in range(iterations):
        CandidateSolution = BestSolution.copy()
        PowerFlow = np.dot(SensitivityMatrix,CandidateSolution)
         

SensivityMatrix = np.array([
    [0.5857, -0.2571, -0.0428, -0.0857, -0.2000],
    [0.2143,  0.0571, -0.1572, -0.1143, -0.0000],
    [0.0905,  0.1619, -0.1953, -0.1238,  0.0666],
    [0.1080,  0.1651, -0.1206, -0.1968,  0.0444],
    [0.1872,  0.2158,  0.0730,  0.0349, -0.5109],
    [0.1048,  0.0191,  0.4476, -0.4380, -0.1334],
    [0.0128, -0.0158,  0.1270,  0.1651, -0.2891]
])

LineLimits = np.array([float(input(f"Enter power limit for Line {i+1}: ")) for i in range(7)])

BusInjection = np.array([float(input(f"Enter power injection for Bus {i+1}: ")) for i in range(5)]).reshape(5, 1)

PowerFlowUnoptimized = np.dot(SensivityMatrix,BusInjection)

BestSolution,FitnessHistory,COptimized = Optimizer(SensivityMatrix,BusInjection,LineLimits)