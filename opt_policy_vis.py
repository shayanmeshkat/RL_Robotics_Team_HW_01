import numpy as np
import environment as env

def extract_optimal_policy(q_values, env_instance):
    """Extract the optimal policy from Q-values"""
    policy = np.zeros((env_instance.grid_x, env_instance.grid_y), dtype=int)
    
    for x in range(env_instance.grid_x):
        for y in range(env_instance.grid_y):
            policy[x, y] = np.argmax(q_values[x, y, :])
    
    return policy

def print_optimal_policy(policy, env_instance):
    """Print the optimal policy in the specified format"""
    # Action symbols mapping
    action_symbols = {
        0: '↑',  # up
        1: '↓',  # down  
        2: '→',  # right
        3: '←'   # left
    }
    
    print("\nOptimal Policy:")
    
    # Print from top to bottom (y=2 to y=0 for a 3-row grid)
    for y in range(env_instance.grid_y - 1, -1, -1):
        row_str = f"y={y+1}: "
        
        for x in range(env_instance.grid_x):
            # Check if this is the starting position
            if (x, y) == (0, 0):
                row_str += "S"
            # Check if this is a wall/block (position 1,1 based on environment)
            elif (x, y) == (1, 1):
                row_str += "XX"
            # Check if this is a goal location
            elif (x, y) in env_instance.goal_locations:
                row_str += "+1"
            # Check if this is a penalty location  
            elif (x, y) in env_instance.penalty_locations:
                row_str += "-1"
            else:
                # Regular cell - show optimal action
                optimal_action = policy[x, y]
                row_str += action_symbols[optimal_action]
            
            # Add space between cells (except last one)
            if x < env_instance.grid_x - 1:
                row_str += " "
        
        print(row_str)
    
    # Print x-axis labels
    x_labels = "x=" + " ".join([str(i+1) for i in range(env_instance.grid_x)])
    print(x_labels)
    print()