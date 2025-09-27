import matplotlib
# Don't set backend here - let display_config handle it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import display_config

def plot_q_values(q_values, env_instance, iteration=None, save_path="./Results"):
    """
    Plot Q-values in a grid format showing all action values for each cell with optimal policy arrows.
    
    Args:
        q_values: numpy array of shape (grid_x, grid_y, num_actions)
        env_instance: environment instance with map information
        iteration: iteration number for filename
        save_path: directory to save the plot
    """
    show_plots, save_plots, interactive_mode = display_config.get_visualization_flags()
    
    # Ensure save directory exists
    if save_plots:
        os.makedirs(save_path, exist_ok=True)
    
    # Create figure with larger size for presentation
    fig, ax = plt.subplots(figsize=(16, 12))
    
    grid_x, grid_y = env_instance.grid_x, env_instance.grid_y
    
    # Draw grid lines with thicker lines
    for i in range(grid_x + 1):
        ax.plot([i, i], [0, grid_y], 'k-', alpha=0.5, linewidth=2)
    for j in range(grid_y + 1):
        ax.plot([0, grid_x], [j, j], 'k-', alpha=0.5, linewidth=2)
    
    # Draw walls (blocked cells)
    wall_rect = patches.Rectangle((1, 1), 1, 1, linewidth=3, edgecolor='k', 
                                 facecolor='gray', alpha=0.9)
    ax.add_patch(wall_rect)
    ax.text(1.5, 1.5, 'WALL', ha='center', va='center', 
            color='white', fontsize=16, fontweight='bold')
    
    # Draw goal locations with light background
    for pos in env_instance.goal_locations:
        x, y = pos
        goal_rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='green', 
                                     facecolor='lightgreen', alpha=0.4)
        ax.add_patch(goal_rect)
    
    # Draw penalty locations with light background
    for pos in env_instance.penalty_locations:
        x, y = pos
        penalty_rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='red', 
                                       facecolor='lightcoral', alpha=0.4)
        ax.add_patch(penalty_rect)
    
    # Action names and positions for display
    action_names = ['↑', '↓', '→', '←']  # up, down, right, left
    action_positions = [(0.5, 0.85), (0.5, 0.15), (0.85, 0.5), (0.15, 0.5)]  # moved closer to edges
    
    # Arrow directions for optimal policy
    arrow_directions = {
        0: (0, 0.2),    # up (reduced from 0.3)
        1: (0, -0.2),   # down (reduced from -0.3)
        2: (0.2, 0),    # right (reduced from 0.3)
        3: (-0.2, 0)    # left (reduced from -0.3)
    }
    
    # Plot Q-values and optimal policy for each cell
    for x in range(grid_x):
        for y in range(grid_y):
            # Skip wall cells
            if (x, y) == (1, 1):
                continue
                
            # Get Q-values for this state
            q_vals = q_values[x, y, :]
            
            # Find optimal action (max Q-value)
            optimal_action = np.argmax(q_vals)
            max_q = np.max(q_vals)
            
            # Draw optimal policy arrow (smaller to avoid overlap)
            if max_q != 0:  # Only draw arrow if there are learned Q-values
                arrow_dx, arrow_dy = arrow_directions[optimal_action]
                arrow_start_x = x + 0.5
                arrow_start_y = y + 0.5
                
                # Draw smaller arrow for optimal policy
                ax.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy,
                        head_width=0.1, head_length=0.08, fc='blue', ec='blue',
                        linewidth=4, alpha=0.8, length_includes_head=True)
            
            # Display each action's Q-value with larger fonts
            for action in range(len(q_vals)):
                rel_x, rel_y = action_positions[action]
                text_x = x + rel_x
                text_y = y + rel_y
                
                q_val = q_vals[action]
                
                # Color coding: red for negative, green for positive, bold for max
                if action == optimal_action and max_q != 0:
                    color = 'blue'
                    weight = 'bold'
                    fontsize = 20
                elif q_val < 0:
                    color = 'red'
                    weight = 'normal'
                    fontsize = 16
                elif q_val > 0:
                    color = 'darkgreen'
                    weight = 'normal'
                    fontsize = 16
                else:
                    color = 'black'
                    weight = 'normal'
                    fontsize = 16
                
                # Format Q-value text with larger font
                q_text = f"{action_names[action]}\n{q_val:.2f}"
                
                ax.text(text_x, text_y, q_text, ha='center', va='center',
                       fontsize=fontsize, color=color, fontweight=weight,
                       bbox=dict(boxstyle="round,pad=0.15", facecolor="white", 
                               alpha=0.8, edgecolor='gray'))
    
    # Add special markers for goal and penalty locations with larger fonts
    for pos in env_instance.goal_locations:
        x, y = pos
        ax.text(x + 0.05, y + 0.95, 'GOAL', ha='left', va='top', 
                color='darkgreen', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
    
    for pos in env_instance.penalty_locations:
        x, y = pos
        ax.text(x + 0.05, y + 0.95, 'PIT', ha='left', va='top', 
                color='darkred', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.9))
    
    # Set axis properties
    ax.set_xlim(0, grid_x)
    ax.set_ylim(0, grid_y)
    ax.set_aspect('equal')
    
    # Title and labels with larger fonts for presentation
    if iteration is not None:
        ax.set_title(f'Q-Values and Optimal Policy - Iteration {iteration}', 
                    fontsize=40, fontweight='bold', pad=20)
        filename = f"q_values_iteration_{iteration}.png"
    else:
        ax.set_title('Q-Values and Optimal Policy', fontsize=40, fontweight='bold', pad=20)
        filename = "q_values_final.png"
    
    ax.set_xlabel('X coordinate', fontsize=16, fontweight='bold')
    ax.set_ylabel('Y coordinate', fontsize=16, fontweight='bold')
    
    # Add coordinate labels with larger fonts
    ax.set_xticks(range(grid_x + 1))
    ax.set_yticks(range(grid_y + 1))
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add comprehensive legend with larger fonts
    legend_text = "Legend:\n"
    legend_text += "• Blue Arrows: Optimal Policy Direction\n"
    legend_text += "• Actions: ↑=Up, ↓=Down, →=Right, ←=Left\n"
    legend_text += "• Colors: Blue=Optimal Action, Green=Positive Q-value\n"
    legend_text += "           Red=Negative Q-value, Black=Zero\n"
    legend_text += "• Numbers show Q-values for each action"

    # ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=12,
    #        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", 
    #                 alpha=0.9, edgecolor='orange', linewidth=2),
    #        verticalalignment='bottom', fontweight='bold')
    
    # ax.text(0.02, 0.02, transform=ax.transAxes, fontsize=12,
    #        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", 
    #                 alpha=0.9, edgecolor='orange', linewidth=2),
    #        verticalalignment='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Handle saving and showing based on flags
    if save_plots:
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Q-values plot saved as {filepath}")
    
    if show_plots:
        plt.show()
        if interactive_mode:
            input("Press Enter to continue...")
    
    if not show_plots:
        plt.close()



