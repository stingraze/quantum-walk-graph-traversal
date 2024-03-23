#(C) Tsubasa Kato 3/23/2024 - Quantum walk - Traversal Research - Inspire Search Corporation
#Company Website: https://www.inspiresearch.io/en
#Email: tsubasa@inspiresearch.io
#Created with help from Chat GPT (GPT-4) (OpenAI) and Gemini Advanced (Google)
import pandas as pd
import networkx as nx
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from io import StringIO
from qiskit_aer import Aer

# Sample CSV data
csv_data = """
node_from,node_to
A,B
A,C
B,D
C,D
"""

# Load CSV
df = pd.read_csv(StringIO(csv_data))

# Construct graph 
G = nx.from_pandas_edgelist(df, source='node_from', target='node_to')

# Map nodes to qubits (integer representation)
node_to_qubit = {node: i for i, node in enumerate(G.nodes)}

# Simplified Quantum Walk Circuit with Hadamard Shift
def graph_traversal_circuit(G, node_to_qubit):
    num_qubits = len(G.nodes)
    circuit = QuantumCircuit(num_qubits)

    # Initial uniform superposition
    circuit.h(range(num_qubits))

    # Simplified walk iterations
    for _ in range(1):  # Reducing iterations for simplicity
        circuit.barrier()
        # Apply Hadamard gates based on graph structure
        for edge in G.edges:
            qubit_from = node_to_qubit[edge[0]]
            qubit_to = node_to_qubit[edge[1]]
            # Apply Hadamard to both qubits representing the edge
            circuit.h(qubit_from)
            circuit.h(qubit_to)
        circuit.barrier()
        # Additional Hadamard gates can be applied here if desired

    return circuit

# Prepare and run the circuit
circuit = graph_traversal_circuit(G, node_to_qubit)
circuit.measure_all()

# Use Aer's simulator
simulator = Aer.get_backend('aer_simulator')
compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts()

# Plot the result
plot_histogram(counts)
plt.show()
