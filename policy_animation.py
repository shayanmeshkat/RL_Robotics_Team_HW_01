import matplotlib
# Don't set backend here - let display_config handle it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import environment as env
import display_config

def animate_optimal_policy(q_values, env_instance, iteration_num, max_steps=50, save_path="./Results/", is_test=False):
    """
    Animate the agent following the optimal policy and save as gif
    """
    show_plots, save_plots, interactive_mode = display_config.get_visualization_flags()
    
    # Extract optimal policy
    policy = np.zeros((env_instance.grid_x, env_instance.grid_y), dtype=int)
    for x in range(env_instance.grid_x):
        for y in range(env_instance.grid_y):
            policy[x, y] = np.argmax(q_values[x, y, :])
    
    # Simulate agent following optimal policy
    agent_path = []
    rewards_path = []
    
    # Reset environment
    test_env = env.env()
    test_env.map()
    state_x, state_y = test_env.agent
    agent_path.append((state_x, state_y))
    
    # Initial reward is 0 (no action taken yet)
    rewards_path.append(0)
    
    # Follow optimal policy
    for step in range(max_steps):
        # Choose optimal action (greedy)
        action = np.argmax(q_values[state_x, state_y, :])
        
        # Execute action (without slip for visualization clarity)
        next_state_x, next_state_y = execute_deterministic_action(state_x, state_y, action, test_env)
        reward, _ = test_env.get_reward()
        
        agent_path.append((next_state_x, next_state_y))
        rewards_path.append(reward)
        
        state_x, state_y = next_state_x, next_state_y
        
        # Stop if reached goal or penalty
        if reward == 1 or reward == -1:
            break
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate_frame(frame):
        ax.clear()
        
        # Draw grid
        for i in range(env_instance.grid_x + 1):
            ax.plot([i, i], [0, env_instance.grid_y], 'k-', alpha=0.3)
        for j in range(env_instance.grid_y + 1):
            ax.plot([0, env_instance.grid_x], [j, j], 'k-', alpha=0.3)
        
        # Draw wall at (1,1)
        wall_rect = patches.Rectangle((1, 1), 1, 1, linewidth=2, edgecolor='k', 
                                     facecolor='black', alpha=0.8)
        ax.add_patch(wall_rect)
        ax.text(1.5, 1.5, 'Wall', ha='center', va='center', 
                color='white', fontsize=10, fontweight='bold')
        
        # Draw goal locations
        for pos in env_instance.goal_locations:
            x, y = pos
            goal_rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='green', 
                                         facecolor='lightgreen', alpha=0.7)
            ax.add_patch(goal_rect)
            ax.text(x + 0.5, y + 0.5, 'GOAL', ha='center', va='center', 
                    color='darkgreen', fontsize=10, fontweight='bold')
        
        # Draw penalty locations
        for pos in env_instance.penalty_locations:
            x, y = pos
            penalty_rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='red', 
                                           facecolor='lightcoral', alpha=0.7)
            ax.add_patch(penalty_rect)
            ax.text(x + 0.5, y + 0.5, 'PIT', ha='center', va='center', 
                    color='darkred', fontsize=10, fontweight='bold')
        
        # Draw path up to current frame
        if frame > 0:
            path_x = [pos[0] + 0.5 for pos in agent_path[:frame+1]]
            path_y = [pos[1] + 0.5 for pos in agent_path[:frame+1]]
            ax.plot(path_x, path_y, 'b--', alpha=0.6, linewidth=2, label='Path')
        
        # Draw agent at current position
        if frame < len(agent_path):
            agent_x, agent_y = agent_path[frame]
            agent_circle = plt.Circle((agent_x + 0.5, agent_y + 0.5), 0.3, 
                                     facecolor='blue', edgecolor='navy', linewidth=3)
            ax.add_patch(agent_circle)
            ax.text(agent_x + 0.5, agent_y + 0.5, 'A', ha='center', va='center', 
                    color='white', fontsize=12, fontweight='bold')
        
        # Set axis properties
        ax.set_xlim(0, env_instance.grid_x)
        ax.set_ylim(0, env_instance.grid_y)
        ax.set_aspect('equal')
        
        animation_type = "Testing" if is_test else "Training"
        ax.set_title(f'Optimal Policy Animation - {animation_type} Iteration {iteration_num}\nStep {frame}/{len(agent_path)-1}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        # Add coordinate labels
        ax.set_xticks(range(env_instance.grid_x + 1))
        ax.set_yticks(range(env_instance.grid_y + 1))
        
        # Add step info - current reward is from this step
        if frame < len(rewards_path):
            current_reward = rewards_path[frame]
            # Total reward is sum of all rewards up to current step (excluding initial 0)
            total_reward = sum(rewards_path[1:frame+1]) if frame > 0 else 0
            
            reward_text = f'Current Reward: {current_reward:.3f}'
            reward_text += f'\nTotal Reward: {total_reward:.3f}'
            ax.text(0.02, 0.98, reward_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   verticalalignment='top')
    
    # Create animation with better error handling
    try:
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(agent_path), 
                                      interval=800, repeat=True, blit=False)
        
        # Save animation
        if save_plots:
            animation_type = "test" if is_test else "train"
            filename = f"{save_path}optimal_policy_animation_{animation_type}_iter_{iteration_num}.gif"
            anim.save(filename, writer='pillow', fps=1.2)
            print(f"Animation saved as {filename}")
        
        # Show animation if requested
        if show_plots:
            plt.show()
            if interactive_mode:
                input("Press Enter to continue...")
                
    except Exception as e:
        print(f"Warning: Could not create animation - {e}")
        if save_plots:
            animation_type = "test" if is_test else "train"
            filename = f"{save_path}optimal_policy_animation_{animation_type}_iter_{iteration_num}_failed.txt"
            with open(filename, 'w') as f:
                f.write(f"Animation creation failed: {e}")
    finally:
        if not show_plots:
            plt.close()
    
    return agent_path, rewards_path

def execute_deterministic_action(state_x, state_y, action, env_instance):
    """Execute action deterministically (no slip) for cleaner visualization"""
    x, y = state_x, state_y
    
    # Map action numbers to enum values
    action_enum = env.ActionsNew(action)
    
    # Check if action is forbidden
    if (x, y, action_enum) not in env_instance.forbidden_transitions:
        if action_enum == env.ActionsNew.down:  # down
            y = y - 1
        elif action_enum == env.ActionsNew.left:  # left
            x = x - 1
        elif action_enum == env.ActionsNew.right:  # right
            x = x + 1
        elif action_enum == env.ActionsNew.up:  # up
            y = y + 1
    
    # Update environment agent position
    env_instance.agent = (x, y)
    return x, y

def simulate_test_episode(q_values, env_instance, max_steps=50):
    """Simulate a single test episode and return the path and rewards"""
    agent_path = []
    rewards_path = []
    
    # Reset environment
    test_env = env.env()
    test_env.map()
    state_x, state_y = test_env.agent
    agent_path.append((state_x, state_y))
    
    # Initial reward is 0 (no action taken yet)
    rewards_path.append(0)
    
    # Follow optimal policy with slip
    for step in range(max_steps):
        # Choose optimal action (greedy)
        action = np.argmax(q_values[state_x, state_y, :])
        
        # Execute action with slip (like in actual testing)
        next_state_x, next_state_y = test_env.execute_action(state_x, state_y, action)
        reward, terminal_condition = test_env.get_reward()
        
        agent_path.append((next_state_x, next_state_y))
        rewards_path.append(reward)
        
        state_x, state_y = next_state_x, next_state_y
        
        # Stop if reached goal or penalty
        if terminal_condition:
            break
    
    return agent_path, rewards_path
      

def simulate_test_episode(q_values, env_instance, max_steps=50):
    """Simulate a single test episode and return the path and rewards"""
    agent_path = []
    rewards_path = []
    
    # Reset environment
    test_env = env.env()
    test_env.map()
    state_x, state_y = test_env.agent
    agent_path.append((state_x, state_y))
    
    # Initial reward is 0 (no action taken yet)
    rewards_path.append(0)
    
    # Follow optimal policy with slip
    for step in range(max_steps):
        # Choose optimal action (greedy)
        action = np.argmax(q_values[state_x, state_y, :])
        
        # Execute action with slip (like in actual testing)
        next_state_x, next_state_y = test_env.execute_action(state_x, state_y, action)
        reward, terminal_condition = test_env.get_reward()
        
        agent_path.append((next_state_x, next_state_y))
        rewards_path.append(reward)
        
        state_x, state_y = next_state_x, next_state_y
        
        # Stop if reached goal or penalty
        if terminal_condition:
            break
    
    return agent_path, rewards_path
