import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Load data from CSV files (using raw string paths to avoid escape sequence issues)
bus_data = pd.read_csv(r"D:/code/conf/load_bus_shunt_capacitor_data.csv")  
generator_data = pd.read_csv(r"D:/code/conf/generator_data.csv")

# Debugging: Print column names to check correctness
print("Bus Data Columns:", bus_data.columns)
print("Generator Data Columns:", generator_data.columns)

# Ensure column names are stripped of extra spaces
bus_data.columns = bus_data.columns.str.strip()
generator_data.columns = generator_data.columns.str.strip()

# Create a power grid graph
G = nx.Graph()

# Define bus connectivity (edges)
edges = [
    (1,2),(1,3),(2,4),(2,5),(2,6),(3,4),(4,6),(4,12),(4,13),(5,7),(6,9),(6,7),(6,8),
    (6,28),(6,10),(8,28),(9,10),(9,11),(10,20),(10,17),
    (10,21),(10,22),(21,22),(12,14),(12,15),(12,16),(14,15),(15,18),(16,17),(18,19),
    (19,20),(22,24),(23,15),(23,24),(24,25),(25,26),(25,27),(27,28),(27,29),(27,30),
    (29,30)
]
G.add_edges_from(edges)

# Assign node attributes from the CSV data
for _, row in bus_data.iterrows():
    bus_id = row.get('Bus No', None)  # Get Bus ID safely
    if pd.notna(bus_id) and bus_id in G.nodes:
        G.nodes[bus_id]['Shunt Capacitor'] = row.get('Shunt Capacitor', 'N/A')
        G.nodes[bus_id]['Load MW'] = row.get('Load MW', 'N/A')
        G.nodes[bus_id]['Load MVar'] = row.get('Load MVar', 'N/A')

for _, row in generator_data.iterrows():
    gen_id = row.get('Gen No', None)  # Get Generator ID safely
    if pd.notna(gen_id) and gen_id in G.nodes:
        G.nodes[gen_id]['Pmin'] = row.get('Pmin', 'N/A')
        G.nodes[gen_id]['Pmax'] = row.get('Pmax', 'N/A')
        G.nodes[gen_id]['Qmin'] = row.get('Qmin', 'N/A')
        G.nodes[gen_id]['Qmax'] = row.get('Qmax', 'N/A')

# Create a position layout for visualization
pos = nx.spring_layout(G, seed=42)

# Extract node positions and attributes for hover text
node_x, node_y, node_text = [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

    # Generate hover text with load values included
    attributes = G.nodes[node]
    hover_text = f"Bus: {node}<br>" + "<br>".join(f"{k}: {v}" for k, v in attributes.items())
    node_text.append(hover_text)

# Extract edge positions
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# Create the figure
fig = go.Figure()

# Add edges
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='gray'),
    hoverinfo='none',
    mode='lines'
))

# Add nodes
fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    marker=dict(size=10, color='blue'),
    text=[str(n) for n in G.nodes()],
    hoverinfo='text',
    hovertext=node_text
))

# Configure layout
fig.update_layout(
    title="Interactive Power Grid Network (with Load Data)",
    showlegend=False,
    hovermode='closest'
)

# Show the interactive plot
fig.show()
