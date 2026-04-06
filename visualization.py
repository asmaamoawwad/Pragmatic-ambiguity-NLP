import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Configure aesthetics to match University/Research standards
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white'})

class PragmaticEvaluationVisualizer:
    """
    A suite of visualization tools for evaluating Pragmatic Ambiguity in NLP.
    This class generates charts based on Semantic Displacement Vector (SDV) analysis.
    """

    @staticmethod
    def plot_accuracy_cascade(results_dict=None):
        """Generates the Waterfall/Cascade chart showing performance degradation."""
        if results_dict is None:
            # Default values from the research poster if no dynamic data is provided
            results_dict = {'Syntax': 96, 'Semantics': 92, 'Literal': 84, 'Pragmatic': 42}
        
        stages = list(results_dict.keys())
        values = list(results_dict.values())
        colors = ["#2E75B6", "#5B9BD5", "#A5A5A5", "#ED7D31"]

        plt.figure(figsize=(9, 6))
        bars = plt.bar(stages, values, color=colors, width=0.6, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 2, f"{height}%", 
                     ha="center", fontweight="bold", color="#333333")

        plt.title("The Accuracy Cascade: Performance Drop-off at Pragmatic Levels", pad=20)
        plt.ylabel("Accuracy Percentage (%)")
        plt.ylim(0, 110)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_semantic_displacement_3d():
        """Visualizes the relationship between complexity and model understanding."""
        x = np.linspace(0, 10, 100) # Contextual Complexity
        y = np.linspace(0, 10, 100) # Model Scale (Parameters)
        X, Y = np.meshgrid(x, y)
        
        # Mathematical representation of the Pragmatic Gap
        Z = 94 - (X**1.8) + (np.log1p(Y) * 3)
        Z = np.clip(Z, 40, 96)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='RdYlBu_r', edgecolor='none', alpha=0.9)
        
        ax.set_title('Geometric Analysis of the Pragmatic Gap', pad=20)
        ax.set_xlabel('Contextual Complexity')
        ax.set_ylabel('Model Size')
        ax.set_zlabel('Accuracy (%)')
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.show()

    @staticmethod
    def plot_vector_space_gap(v_literal=None, v_intent=None):
        """Plots the Semantic Displacement Vector (SDV) between literal and pragmatic intent."""
        if v_literal is None: v_literal = np.array([0.8, 0.6])
        if v_intent is None: v_intent = np.array([-0.7, -0.3])

        # Calculate the Gap (The core formula of the research)
        gap_vector = v_intent - v_literal
        gap_magnitude = np.linalg.norm(gap_vector)

        plt.figure(figsize=(8, 8))
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)

        plt.quiver(0, 0, v_literal[0], v_literal[1], angles='xy', scale_units='xy', scale=1, 
                   color='#2E75B6', label='Literal Vector ($V_{lit}$)', width=0.012)
        plt.quiver(0, 0, v_intent[0], v_intent[1], angles='xy', scale_units='xy', scale=1, 
                   color='#ED7D31', label='Intent Vector ($V_{int}$)', width=0.012)
        
        plt.plot([v_literal[0], v_intent[0]], [v_literal[1], v_intent[1]], 
                 'g--', lw=2, label=f'SDV Gap (||Δ|| = {gap_magnitude:.2f})')
        
        plt.title("Vector Space Analysis: Semantic Displacement", pad=15)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # Create the visualizer instance
    viz = PragmaticEvaluationVisualizer()
    
    # Run the visualization suite
    viz.plot_accuracy_cascade()
    viz.plot_semantic_displacement_3d()
    viz.plot_vector_space_gap()
