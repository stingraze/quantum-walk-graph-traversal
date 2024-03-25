#(C)Tsubasa Kato 2024/3/25 16:02PM JST
#Purely experimental, and not fully working. Coded with help of Google Gemini Advanced & OpenAI ChatGPT (GPT-4)
import requests
from bs4 import BeautifulSoup
import pandas as pd
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit_aer import Aer

# Function to simulate graph traversal using a quantum circuit
def graph_traversal_circuit(G, node_to_qubit):
    num_qubits = len(G.nodes)
    circuit = QuantumCircuit(num_qubits)

    # Initial uniform superposition
    circuit.h(range(num_qubits))

    # Simplified walk iterations
    for _ in range(100):  # Adjust the number of iterations if necessary
        circuit.barrier()
        # Apply Hadamard gates based on graph structure
        for edge in G.edges:
            qubit_from = node_to_qubit[edge[0]]
            qubit_to = node_to_qubit[edge[1]]
            circuit.h(qubit_from)
            circuit.h(qubit_to)
        circuit.barrier()

    return circuit

# Modified function to crawl and construct a graph with limited size
def crawl_and_construct_graph(urls_file, max_nodes=29):
    with open(urls_file, 'r') as f:
        urls = f.read().splitlines()

    edges = []
    node_count = 0
    for url in urls:
        if node_count >= max_nodes:
            break

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Simplify the linkage criterion for demonstration
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.startswith('http'):
                    print("Found Link: " + href)    
                    edges.append((url, href))
                    node_count += 1
                    if node_count >= max_nodes:
                        break
                else:
                    print("Link not found.")

        except Exception as e:
            print(f"Failed to fetch or parse {url}: {e}")

    df = pd.DataFrame(edges, columns=['node_from', 'node_to'])
    return df

# Main logic
if __name__ == '__main__':
    df = crawl_and_construct_graph('urls.txt', max_nodes=29)
    G = nx.from_pandas_edgelist(df, source='node_from', target='node_to')

    # Mapping nodes to qubits
    node_to_qubit = {node: i for i, node in enumerate(G.nodes)}

    # Prepare and run the quantum circuit for graph traversal
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
