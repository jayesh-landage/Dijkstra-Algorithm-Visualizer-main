import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import networkx as nx
import math
import heapq

class GraphLearningPlatform:
    def __init__(self, master):
        self.master = master
        self.master.title("Dijkstra Algorithm Visualizer")
        self.master.geometry("1200x800")
        
        self.bg_color = '#1e1e2e'
        self.fg_color = '#cdd6f4'
        self.accent_color = '#89b4fa'
        self.success_color = '#a6e3a1'
        self.warning_color = '#fab387'
        
        self.master.configure(bg=self.bg_color)
        
        self.graph = nx.Graph()
        self.pos = {}
        
        self.animation_running = False
        self.current_step = 0
        self.animation_steps = []
        self.visited_nodes = set()
        self.distances = {}
        self.final_path = []
        self.final_distance = 0

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', background=self.bg_color, foreground=self.accent_color, 
                       font=('Helvetica', 20, 'bold'))
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color, 
                       font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10, 'bold'), padding=5)
        style.configure('TEntry', fieldbackground='#313244', foreground=self.fg_color)
        style.configure('TFrame', background=self.bg_color)

        self.title_label = ttk.Label(master, text="🔷 Dijkstra Algorithm Visualizer 🔷", 
                                     style='Title.TLabel')
        self.title_label.pack(pady=15)

        main_container = ttk.Frame(master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        input_container = ttk.LabelFrame(left_panel, text=" Graph Construction ", padding=10)
        input_container.pack(pady=10, fill=tk.X)

        ttk.Label(input_container, text="Node Name:").grid(row=0, column=0, sticky='w', pady=5)
        self.node_entry = ttk.Entry(input_container, width=20)
        self.node_entry.grid(row=0, column=1, pady=5, padx=5)
        
        self.add_node_button = tk.Button(input_container, text="➕ Add Node", 
                                         command=self.add_node, bg=self.success_color, 
                                         fg='black', font=('Helvetica', 10, 'bold'),
                                         relief=tk.FLAT, cursor='hand2')
        self.add_node_button.grid(row=0, column=2, padx=5)

        ttk.Label(input_container, text="Edge (A,B,weight):").grid(row=1, column=0, sticky='w', pady=5)
        self.edge_entry = ttk.Entry(input_container, width=20)
        self.edge_entry.grid(row=1, column=1, pady=5, padx=5)
        
        self.add_edge_button = tk.Button(input_container, text="➕ Add Edge", 
                                         command=self.add_edge, bg=self.success_color,
                                         fg='black', font=('Helvetica', 10, 'bold'),
                                         relief=tk.FLAT, cursor='hand2')
        self.add_edge_button.grid(row=1, column=2, padx=5)

        algo_container = ttk.LabelFrame(left_panel, text=" Algorithm Controls ", padding=10)
        algo_container.pack(pady=10, fill=tk.X)

        ttk.Label(algo_container, text="Source Node:").grid(row=0, column=0, sticky='w', pady=5)
        self.source_entry = ttk.Entry(algo_container, width=15)
        self.source_entry.grid(row=0, column=1, pady=5, padx=5)

        ttk.Label(algo_container, text="Target Node:").grid(row=1, column=0, sticky='w', pady=5)
        self.target_entry = ttk.Entry(algo_container, width=15)
        self.target_entry.grid(row=1, column=1, pady=5, padx=5)

        self.run_button = tk.Button(algo_container, text="▶ Run Animation", 
                                    command=self.run_algorithm_animated,
                                    bg=self.accent_color, fg='black',
                                    font=('Helvetica', 11, 'bold'),
                                    relief=tk.FLAT, cursor='hand2', width=20)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.reset_button = tk.Button(algo_container, text="🔄 Reset Visualization", 
                                      command=self.reset_visualization,
                                      bg=self.warning_color, fg='black',
                                      font=('Helvetica', 10, 'bold'),
                                      relief=tk.FLAT, cursor='hand2', width=20)
        self.reset_button.grid(row=3, column=0, columnspan=2, pady=5)

        result_container = ttk.LabelFrame(left_panel, text=" Results ", padding=10)
        result_container.pack(pady=10, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(result_container, height=15, width=35, 
                                   bg='#313244', fg=self.fg_color,
                                   font=('Courier', 9), relief=tk.FLAT,
                                   wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(result_container, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plt.style.use('dark_background')
        self.figure, self.ax = plt.subplots(figsize=(10, 8), facecolor=self.bg_color)
        self.ax.set_facecolor(self.bg_color)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.node_entry.bind('<Return>', lambda e: self.add_node())
        self.edge_entry.bind('<Return>', lambda e: self.add_edge())

    def calculate_circular_layout(self):
        """Calculate positions for nodes in a circular layout (clockwise from top)."""
        nodes = list(self.graph.nodes())
        n = len(nodes)
        if n == 0:
            return {}
        
        pos = {}
        radius = 2.0
        
        for i, node in enumerate(nodes):
            angle = math.radians(90 - (i * 360 / n))
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            pos[node] = (x, y)
        
        return pos

    def add_node(self):
        """Add a node to the graph with animation."""
        node = self.node_entry.get().strip()
        if node and node not in self.graph:
            self.graph.add_node(node)
            self.node_entry.delete(0, tk.END)
            self.pos = self.calculate_circular_layout()
            self.visualize_graph()
            self.result_text.insert(tk.END, f"✓ Added node: {node}\n")
            self.result_text.see(tk.END)

    def add_edge(self):
        """Add an edge to the graph."""
        edge_input = self.edge_entry.get().strip().split(',')
        if len(edge_input) == 3:
            node1, node2, weight = [x.strip() for x in edge_input]
            try:
                weight = float(weight)
                if node1 in self.graph and node2 in self.graph:
                    self.graph.add_edge(node1, node2, weight=weight)
                    self.edge_entry.delete(0, tk.END)
                    self.visualize_graph()
                    self.result_text.insert(tk.END, f"✓ Added edge: {node1} ↔ {node2} (weight: {weight})\n")
                    self.result_text.see(tk.END)
                else:
                    self.result_text.insert(tk.END, f"✗ Error: Nodes not found\n")
            except ValueError:
                self.result_text.insert(tk.END, f"✗ Error: Invalid weight\n")

    def visualize_graph(self, highlight_nodes=None, highlight_edges=None, node_labels=None, 
                       current_distances=None, show_final_path=False):
        """Visualize the current graph with straight lines."""
        self.ax.clear()
        if self.graph.number_of_nodes() == 0:
            self.ax.set_title("Graph Visualization", color=self.accent_color, fontsize=14, weight='bold')
            self.canvas.draw()
            return

        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            if show_final_path and highlight_nodes and node in highlight_nodes:
                # Nodes in final shortest path - bright green with larger size
                node_colors.append('#a6e3a1')
                node_sizes.append(1500)
            elif highlight_nodes and node in highlight_nodes:
                if node in self.visited_nodes:
                    node_colors.append('#a6e3a1')  # Visited - green
                else:
                    node_colors.append('#f38ba8')  # Current - pink
                node_sizes.append(1200)
            else:
                node_colors.append('#89b4fa')  # Default - blue
                node_sizes.append(1200)

        edge_colors = []
        edge_widths = []
        edge_styles = []
        for edge in self.graph.edges():
            if show_final_path and highlight_edges and (edge in highlight_edges or (edge[1], edge[0]) in highlight_edges):
                # Edges in final shortest path - bright orange, thicker
                edge_colors.append('#fab387')
                edge_widths.append(6)
                edge_styles.append('solid')
            elif highlight_edges and (edge in highlight_edges or (edge[1], edge[0]) in highlight_edges):
                edge_colors.append('#fab387')  # Currently exploring - orange
                edge_widths.append(4)
                edge_styles.append('solid')
            else:
                edge_colors.append('#6c7086')  # Default - gray
                edge_widths.append(2)
                edge_styles.append('solid')
        
        nx.draw_networkx_nodes(self.graph, self.pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              edgecolors='white',
                              linewidths=3,
                              ax=self.ax)

        nx.draw_networkx_edges(self.graph, self.pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              style=edge_styles,
                              arrows=False,
                              ax=self.ax)

        if node_labels:
            nx.draw_networkx_labels(self.graph, self.pos, 
                                   labels=node_labels,
                                   font_size=9,
                                   font_color='black',
                                   font_weight='bold',
                                   ax=self.ax)
        elif current_distances:
            labels = {}
            for node in self.graph.nodes():
                if node in current_distances:
                    dist = current_distances[node]
                    if dist == float('inf'):
                        labels[node] = f"{node}\n∞"
                    else:
                        labels[node] = f"{node}\n[{dist:.1f}]"
                else:
                    labels[node] = node
            nx.draw_networkx_labels(self.graph, self.pos,
                                   labels=labels,
                                   font_size=9,
                                   font_color='black',
                                   font_weight='bold',
                                   ax=self.ax)
        else:
            nx.draw_networkx_labels(self.graph, self.pos,
                                   font_size=12,
                                   font_color='black',
                                   font_weight='bold',
                                   ax=self.ax)

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, self.pos, 
                                    edge_labels=edge_labels,
                                    font_color='#f9e2af',
                                    font_size=10,
                                    font_weight='bold',
                                    ax=self.ax)

        if show_final_path:
            title = "✓ Shortest Path Found!"
            color = self.success_color
        else:
            title = "Graph Visualization"
            color = self.accent_color
            
        self.ax.set_title(title, color=color, fontsize=14, weight='bold', pad=20)
        self.ax.axis('off')
        self.canvas.draw()

    def dijkstra_algorithm(self, source, target):
        """Implement Dijkstra's algorithm and record steps."""
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[source] = 0
        previous = {node: None for node in self.graph.nodes()}
        pq = [(0, source)]
        visited = set()
        steps = []
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            steps.append({
                'current': current,
                'visited': visited.copy(),
                'distances': distances.copy(),
                'previous': previous.copy()
            })
            
            if current == target:
                break
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    edge_weight = self.graph[current][neighbor]['weight']
                    new_distance = distances[current] + edge_weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
                        heapq.heappush(pq, (new_distance, neighbor))
        
        # Reconstruct path
        path = []
        if distances[target] != float('inf'):  # Path exists
            current = target
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()
        
        return path, distances[target], steps

    def run_algorithm_animated(self):
        """Run Dijkstra's algorithm with step-by-step animation."""
        if self.animation_running:
            return
            
        source = self.source_entry.get().strip()
        target = self.target_entry.get().strip()

        if source not in self.graph or target not in self.graph:
            self.result_text.insert(tk.END, "\n✗ Error: Invalid source or target node\n")
            return

        self.animation_running = True
        self.visited_nodes = set()
        self.current_step = 0
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"🚀 Starting Dijkstra's Algorithm\n")
        self.result_text.insert(tk.END, f"Source: {source} → Target: {target}\n")
        self.result_text.insert(tk.END, "="*40 + "\n\n")

        path, distance, steps = self.dijkstra_algorithm(source, target)
        
        if not path or distance == float('inf'):
            self.result_text.insert(tk.END, "\n✗ No path exists between source and target\n")
            self.animation_running = False
            return
        
        self.animation_steps = steps
        self.final_path = path
        self.final_distance = distance
        self.animate_step()

    def animate_step(self):
        """Animate one step of the algorithm."""
        if self.current_step < len(self.animation_steps):
            step_data = self.animation_steps[self.current_step]
            current_node = step_data['current']
            self.visited_nodes = step_data['visited']
            
            self.visualize_graph(highlight_nodes=self.visited_nodes, 
                               current_distances=step_data['distances'])
            
            self.result_text.insert(tk.END, f"Step {self.current_step + 1}: Visiting '{current_node}' ")
            self.result_text.insert(tk.END, f"(distance: {step_data['distances'][current_node]:.1f})\n")
            self.result_text.see(tk.END)
            
            self.current_step += 1
            self.master.after(1000, self.animate_step)
        else:
            self.show_final_result()

    def show_final_result(self):
        """Display the final result after animation."""
        if not self.final_path:
            self.result_text.insert(tk.END, "\n✗ No path found\n")
            self.animation_running = False
            return
            
        path_edges = list(zip(self.final_path[:-1], self.final_path[1:]))
        
        # Show final path with special highlighting
        self.visualize_graph(highlight_nodes=self.final_path, 
                           highlight_edges=path_edges, 
                           show_final_path=True)
        
        self.result_text.insert(tk.END, "\n" + "="*40 + "\n")
        self.result_text.insert(tk.END, f"✓ Algorithm Complete!\n\n")
        self.result_text.insert(tk.END, f"📏 Shortest Path Distance: {self.final_distance:.2f}\n\n")
        self.result_text.insert(tk.END, f"🛣️  Path Sequence:\n")
        self.result_text.insert(tk.END, f"   {' → '.join(self.final_path)}\n\n")
        
        # Show step-by-step path with distances
        self.result_text.insert(tk.END, f"📊 Path Details:\n")
        cumulative_dist = 0
        for i in range(len(self.final_path) - 1):
            from_node = self.final_path[i]
            to_node = self.final_path[i + 1]
            edge_weight = self.graph[from_node][to_node]['weight']
            cumulative_dist += edge_weight
            self.result_text.insert(tk.END, f"   {from_node} → {to_node}: +{edge_weight:.1f} (total: {cumulative_dist:.1f})\n")
        
        self.result_text.see(tk.END)
        
        self.animation_running = False

    def reset_visualization(self):
        """Reset the visualization."""
        self.animation_running = False
        self.visited_nodes = set()
        self.current_step = 0
        self.animation_steps = []
        self.final_path = []
        self.final_distance = 0
        self.visualize_graph()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Visualization reset. Ready for new algorithm run.\n")

    def on_closing(self):
        """Handle window close event."""
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphLearningPlatform(root)
    root.mainloop()
