import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import pandas as pd
import environment as env
import save_results
from param_study import run_parameter_study, plot_parameter_curves, plot_final_performance_comparison
from opt_policy_vis import extract_optimal_policy, print_optimal_policy
from policy_animation import animate_optimal_policy, create_policy_visualization, simulate_test_episode
from q_val_plot import plot_q_values
import os

training_data_file_name = './Results/base_training_data'
testing_data_file_name = './Results/base_testing_data'



# Set the hyperparameters
alpha = 0.01  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate
max_steps = 50  # maximum steps per episode
training_episodes = 5000  # Reduced for parameter study
total_test_episode = 1000
testing_frequency = 10
test_episodes = int(total_test_episode / ((training_episodes - testing_frequency)//testing_frequency + 1))
num_time = 1  # Reduced for parameter study

# Parameter study ranges - increased differences for clearer effects
alpha_values = [0.01, 0.1, 0.5, 0.9]
gamma_values = [0.1, 0.3, 0.9, 0.95]
epsilon_values = [0.01, 0.1, 0.3, 0.5]



# Initialize the environment class
env_instance = env.env()

# Get the grid size of the environment
grid_x = env_instance.grid_x
grid_y = env_instance.grid_y

# Get the number of actions in the environment
num_actions = env_instance.actions_num

# Initialize the q_values array
q_values = np.zeros((grid_x, grid_y, num_actions))

# Define the Q-learning algorithm
def q_learning(alpha, gamma, epsilon, max_steps, episodes):
    rewards_per_episode = []
    for episode in range(episodes):
        total_reward = 0
        env_instance = env.env()
        # rm_state = rm.get_initial_state()
        reward_rm_sum = 0
        state_x, state_y = env_instance.agent
        env_instance.map()
        # event_1_flag, event_2_flag, event_3_flag, event_4_flag = 0, 0, 0, 0

        for step in range(max_steps):
            action = env_instance.choose_action(q_values[state_x, state_y, :], state_x, state_y, epsilon)

            

            next_state_x, next_state_y = env_instance.execute_action(state_x, state_y, action)

            reward = env_instance.get_reward()
            # event = env_instance.get_true_propositions(step)
            # rm_next_state = rm.get_next_state(rm_state, event)
            # reward_rm = rm.get_reward(rm_state, rm_next_state)
            reward_rm_sum += reward


            total_reward += reward
            q_values[state_x, state_y, action] += alpha * (reward + gamma * np.max(q_values[next_state_x, next_state_y, :]) - q_values[state_x, state_y, action])
            
            state_x = next_state_x
            state_y = next_state_y

            if reward == 1 or reward == -100:  # If all rewards obtained, break
                break

        rewards_per_episode.append(total_reward)
    return rewards_per_episode

# Testing function
def test_policy(q_values, test_episodes, max_steps):
    rewards_per_episode = []
    for episode in range(test_episodes):
        total_reward = 0
        env_instance = env.env()
        
        reward_rm_sum = 0
        state_x, state_y = env_instance.agent
        env_instance.map()
        

        for step in range(max_steps):
            # Choose action greedily (no exploration)
            action = np.argmax(q_values[state_x, state_y, :])

            next_state_x, next_state_y = env_instance.execute_action(state_x, state_y, action)
            
            reward = env_instance.get_reward()
            
            reward_rm_sum += reward

            

            total_reward += reward
            
            state_x = next_state_x
            state_y = next_state_y

            if reward == 1 or reward == -100:  # If all rewards obtained, break
                break

        rewards_per_episode.append(total_reward)
    return rewards_per_episode



# Create Results directory if it doesn't exist
os.makedirs('./Results', exist_ok=True)

# Run Q-learning and test the policy
reward_list = [[] for _ in range(num_time)]
test_reward_list = [[] for _ in range(num_time)]

for iter in range(num_time):
    np.random.seed(iter)
    rewards = []
    test_rewards = []
    
    print(f"\n=== Training Iteration {iter + 1}/{num_time} ===")
    
    for episode in range(training_episodes):
        rewards.extend(q_learning(alpha, gamma, epsilon, max_steps, 1))
        if (episode + 1) % testing_frequency == 0:
            test_rewards.extend(test_policy(q_values, test_episodes, max_steps))
    
    # Print optimal policy after training iteration
    env_temp = env.env()
    env_temp.map()
    optimal_policy = extract_optimal_policy(q_values, env_temp)
    print_optimal_policy(optimal_policy, env_temp)
    
    # Plot Q-values at the end of each iteration
    print(f"Creating Q-values visualization for iteration {iter + 1}...")
    plot_q_values(q_values, env_temp, iteration=iter + 1)
    
    # Create training animations and visualizations
    print(f"Creating training animation for iteration {iter + 1}...")
    agent_path, rewards_path = animate_optimal_policy(q_values, env_temp, iter + 1, is_test=False)
    create_policy_visualization(q_values, env_temp, iter + 1, is_test=False)
    
    print(f"Training - Agent path: {agent_path}")
    print(f"Training - Total reward for path: {sum(rewards_path[1:]):.3f}")  # Exclude initial 0
    
    # Create testing animations and visualizations
    print(f"Creating testing animation for iteration {iter + 1}...")
    test_agent_path, test_rewards_path = simulate_test_episode(q_values, env_temp, max_steps)
    
    # Create testing animation using the simulated test episode
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate_test_frame(frame):
        ax.clear()
        
        # Draw grid
        for i in range(env_temp.grid_x + 1):
            ax.plot([i, i], [0, env_temp.grid_y], 'k-', alpha=0.3)
        for j in range(env_temp.grid_y + 1):
            ax.plot([0, env_temp.grid_x], [j, j], 'k-', alpha=0.3)
        
        # Draw wall at (1,1)
        wall_rect = patches.Rectangle((1, 1), 1, 1, linewidth=2, edgecolor='k', 
                                     facecolor='black', alpha=0.8)
        ax.add_patch(wall_rect)
        ax.text(1.5, 1.5, 'Wall', ha='center', va='center', 
                color='white', fontsize=10, fontweight='bold')
        
        # Draw goal and penalty locations
        for pos in env_temp.goal_locations:
            x, y = pos
            goal_rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='green', 
                                         facecolor='lightgreen', alpha=0.7)
            ax.add_patch(goal_rect)
            ax.text(x + 0.5, y + 0.5, 'GOAL', ha='center', va='center', 
                    color='darkgreen', fontsize=10, fontweight='bold')
        
        for pos in env_temp.penalty_locations:
            x, y = pos
            penalty_rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='red', 
                                           facecolor='lightcoral', alpha=0.7)
            ax.add_patch(penalty_rect)
            ax.text(x + 0.5, y + 0.5, 'PIT', ha='center', va='center', 
                    color='darkred', fontsize=10, fontweight='bold')
        
        # Draw path up to current frame
        if frame > 0:
            path_x = [pos[0] + 0.5 for pos in test_agent_path[:frame+1]]
            path_y = [pos[1] + 0.5 for pos in test_agent_path[:frame+1]]
            ax.plot(path_x, path_y, 'b--', alpha=0.6, linewidth=2, label='Path')
        
        # Draw agent at current position
        if frame < len(test_agent_path):
            agent_x, agent_y = test_agent_path[frame]
            agent_circle = plt.Circle((agent_x + 0.5, agent_y + 0.5), 0.3, 
                                     color='blue', linewidth=3, edgecolor='navy')
            ax.add_patch(agent_circle)
            ax.text(agent_x + 0.5, agent_y + 0.5, 'A', ha='center', va='center', 
                    color='white', fontsize=12, fontweight='bold')
        
        # Set axis properties
        ax.set_xlim(0, env_temp.grid_x)
        ax.set_ylim(0, env_temp.grid_y)
        ax.set_aspect('equal')
        ax.set_title(f'Testing Episode Animation - Iteration {iter + 1}\nStep {frame}/{len(test_agent_path)-1}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_xticks(range(env_temp.grid_x + 1))
        ax.set_yticks(range(env_temp.grid_y + 1))
        
        # Add step info
        if frame < len(test_rewards_path):
            current_reward = test_rewards_path[frame]
            total_reward = sum(test_rewards_path[1:frame+1]) if frame > 0 else 0
            
            reward_text = f'Current Reward: {current_reward:.3f}'
            reward_text += f'\nTotal Reward: {total_reward:.3f}'
            ax.text(0.02, 0.98, reward_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                   verticalalignment='top')
    
    # Create and save testing animation
    test_anim = animation.FuncAnimation(fig, animate_test_frame, frames=len(test_agent_path), 
                                       interval=800, repeat=True, blit=False)
    
    test_filename = f"./Results/testing_episode_animation_iter_{iter + 1}.gif"
    test_anim.save(test_filename, writer='pillow', fps=1.2)
    plt.close()
    
    create_policy_visualization(q_values, env_temp, iter + 1, is_test=True)
    
    print(f"Testing animation saved as {test_filename}")
    print(f"Testing - Agent path: {test_agent_path}")
    print(f"Testing - Total reward for path: {sum(test_rewards_path[1:]):.3f}")  # Exclude initial 0
    
    reward_list[iter] = rewards
    test_reward_list[iter] = test_rewards

    # Reset q-values for next iteration to avoid variance issues
    q_values = np.zeros((grid_x, grid_y, num_actions))

# Save the data as a parquet file
save_results.save_reward(reward_list, training_data_file_name)
save_results.save_reward(test_reward_list, testing_data_file_name)

# Plot the training rewards
rolling_mean_window = 10
rolling_mean = pd.Series(rewards).rolling(window=rolling_mean_window).mean()
rolling_mean_test = pd.Series(test_rewards).rolling(window=rolling_mean_window).mean()
rew = np.array(rolling_mean.fillna(0))
rew_test = np.array(rolling_mean_test.fillna(0))

plt.plot(range(len(rolling_mean)), rew,
         label=f'Rolling Mean (window={rolling_mean_window})',
         color='blue')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Reward per Episode')
plt.legend()
plt.grid('minor')
plt.show()

# Plot the testing rewards
plt.plot(range(len(rolling_mean_test)), rew_test, label='Testing Rewards', color='red')
plt.xlabel('Test Run')
plt.ylabel('Average Reward')
plt.title('Average Testing Reward per Run')
plt.legend()
plt.grid('minor')
plt.show()



# Run parameter study
# print("Starting Parameter Study...")
# alpha_curves, gamma_curves, epsilon_curves = run_parameter_study(alpha_values, gamma_values, epsilon_values, training_episodes, testing_frequency, test_episodes, max_steps, num_time, q_values, epsilon, alpha, gamma, grid_x, grid_y, num_actions)
# plot_parameter_curves(alpha_curves, gamma_curves, epsilon_curves)
# plot_final_performance_comparison(alpha_curves, gamma_curves, epsilon_curves)


