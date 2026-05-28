import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_functions():
    # Set x between -3 and 3
    x = np.linspace(-3, 3, 1000)
    
    # Calculate functions
    y_sigmoid = sigmoid(x)
    y_log_sigmoid = np.log(y_sigmoid)
    y_neg_log_sigmoid = -np.log(y_sigmoid)
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
    # Subplot 1: sigmoid(x)
    axs[0].plot(x, y_sigmoid, color='blue', linewidth=2)
    axs[0].set_title('sigmoid(x)')
    axs[0].set_ylabel('y')
    axs[0].grid(True, alpha=0.3)
    axs[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Subplot 2: log(sigmoid(x))
    axs[1].plot(x, y_log_sigmoid, color='green', linewidth=2, linestyle='--')
    axs[1].set_title('log(sigmoid(x))')
    axs[1].set_ylabel('y')
    axs[1].grid(True, alpha=0.3)
    axs[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Subplot 3: -log(sigmoid(x))
    axs[2].plot(x, y_neg_log_sigmoid, color='red', linewidth=2, linestyle='-.')
    axs[2].set_title('-log(sigmoid(x))')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].grid(True, alpha=0.3)
    axs[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'sigmoid_plots.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_functions()
