import networkx as nx
import matplotlib.pyplot as plt

def create_power_grid():
    G = nx.Graph()
    
    # Adding edges between buses based on the image
    edges = [
        (1,2),(1,3),(2,4),(2,5),(2,6),(3,4),(4,6),(4,12),(4,13),(5,7),(6,9),(6,7),(6,8),
        (6,28),(6,10),(8,28),(9,10),(9,11),(10,20),(10,17),
        (10,21),(10,22),(21,22),(12,14),(12,15),(12,16),(14,15),(15,18),(16,17),(18,19),
        (19,20),(22,24),(23,15),(23,24),(24,25),(25,26),(25,27),(27,28),(27,29),(27,30),
        (29,30)
    ]
    
    G.add_edges_from(edges)
    
    # Define positions manually for better alignment based on the schematic diagram
    pos = {
        1: (0, 5), 2: (1, 5), 3: (0, 4.5), 4: (1, 4), 5: (2, 5), 6: (2, 4), 7: (3, 5), 8: (3, 2.5),
        9: (2, 3.5), 10: (2, 3), 11: (1.5, 2.5), 12: (1, 3.5), 13: (0.5, 3.8), 14: (0.8, 3), 15: (1.5, 3),
        16: (2, 2.7), 17: (2.5, 2.5), 18: (2, 2.2), 19: (2.8, 2.2), 20: (3, 2), 21: (3.5, 2.5), 22: (3.5, 2),
        23: (3, 3.2), 24: (3.8, 3), 25: (4.2, 3.5), 26: (4.2, 2.5), 27: (4.5, 3), 28: (4.5, 2), 29: (5, 3.5),
        30: (5, 2.5)
    }
    
    # Drawing the power grid
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color='teal',font_color="lightgray",edge_color='darkgray', node_size=700, font_size=10)
    plt.title("Power Grid Bus Connections - Aligned Layout")
    plt.show()
    
    return G

# Example usage
power_grid = create_power_grid()
