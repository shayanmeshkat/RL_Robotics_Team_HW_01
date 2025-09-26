import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import environment as env
def run_parameter_study(alpha_values, gamma_values, epsilon_values, training_episodes, testing_frequency, test_episodes, max_steps, num_time, q_values, epsilon, alpha, gamma, grid_x, grid_y, num_actions):
    """Run parameter study for alpha, gamma, and epsilon with reward curves"""
    
    # Study learning rate (alpha)
    print("Running parameter study for learning rate (alpha)...")
    alpha_curves = {}
    for alpha_val in alpha_values:
        print(f"Testing alpha = {alpha_val}")
        all_rewards = []
        
        for iter in range(num_time):
            np.random.seed(iter)  # Different seeds for each parameter
            q_values_local = np.zeros((grid_x, grid_y, num_actions))  # Local Q-values
            
            episode_rewards = []
            for episode in range(training_episodes):
                # Create new environment instance for each episode
                env_temp = env.env()
                total_reward = 0
                state_x, state_y = env_temp.agent
                env_temp.map()
                
                for step in range(max_steps):
                    action = env_temp.choose_action(q_values_local[state_x, state_y, :], state_x, state_y, epsilon)
                    reward = env_temp.get_reward()
                    next_state_x, next_state_y = env_temp.execute_action(state_x, state_y, action)
                    
                    total_reward += reward
                    q_values_local[state_x, state_y, action] += alpha_val * (reward + gamma * np.max(q_values_local[next_state_x, next_state_y, :]) - q_values_local[state_x, state_y, action])
                    
                    state_x = next_state_x
                    state_y = next_state_y
                    
                    if reward == 1 or reward == -1:
                        break
                
                episode_rewards.append(total_reward)
            all_rewards.append(episode_rewards)
        
        # Average across runs
        alpha_curves[alpha_val] = np.mean(all_rewards, axis=0)
    
    # Study discount factor (gamma)
    print("Running parameter study for discount factor (gamma)...")
    gamma_curves = {}
    for gamma_val in gamma_values:
        print(f"Testing gamma = {gamma_val}")
        all_rewards = []
        
        for iter in range(num_time):
            np.random.seed(iter)  # Different seeds
            q_values_local = np.zeros((grid_x, grid_y, num_actions))
            
            episode_rewards = []
            for episode in range(training_episodes):
                env_temp = env.env()
                total_reward = 0
                state_x, state_y = env_temp.agent
                env_temp.map()
                
                for step in range(max_steps):
                    action = env_temp.choose_action(q_values_local[state_x, state_y, :], state_x, state_y, epsilon)
                    reward = env_temp.get_reward()
                    next_state_x, next_state_y = env_temp.execute_action(state_x, state_y, action)
                    
                    total_reward += reward
                    q_values_local[state_x, state_y, action] += alpha * (reward + gamma_val * np.max(q_values_local[next_state_x, next_state_y, :]) - q_values_local[state_x, state_y, action])
                    
                    state_x = next_state_x
                    state_y = next_state_y
                    
                    if reward == 1 or reward == -1:
                        break
                
                episode_rewards.append(total_reward)
            all_rewards.append(episode_rewards)
        
        gamma_curves[gamma_val] = np.mean(all_rewards, axis=0)
    
    # Study exploration rate (epsilon)
    print("Running parameter study for exploration rate (epsilon)...")
    epsilon_curves = {}
    for epsilon_val in epsilon_values:
        print(f"Testing epsilon = {epsilon_val}")
        all_rewards = []
        
        for iter in range(num_time):
            np.random.seed(iter)  # Different seeds
            q_values_local = np.zeros((grid_x, grid_y, num_actions))
            
            episode_rewards = []
            for episode in range(training_episodes):
                env_temp = env.env()
                total_reward = 0
                state_x, state_y = env_temp.agent
                env_temp.map()
                
                for step in range(max_steps):
                    action = env_temp.choose_action(q_values_local[state_x, state_y, :], state_x, state_y, epsilon_val)
                    reward = env_temp.get_reward()
                    next_state_x, next_state_y = env_temp.execute_action(state_x, state_y, action)
                    
                    total_reward += reward
                    q_values_local[state_x, state_y, action] += alpha * (reward + gamma * np.max(q_values_local[next_state_x, next_state_y, :]) - q_values_local[state_x, state_y, action])
                    
                    state_x = next_state_x
                    state_y = next_state_y
                    
                    if reward == 1 or reward == -1:
                        break
                
                episode_rewards.append(total_reward)
            all_rewards.append(episode_rewards)
        
        epsilon_curves[epsilon_val] = np.mean(all_rewards, axis=0)
    
    return alpha_curves, gamma_curves, epsilon_curves

def plot_parameter_curves(alpha_curves, gamma_curves, epsilon_curves):
    """Plot reward curves for each parameter study"""
    
    rolling_window = 100  # Increased for smoother curves
    
    # Plot alpha curves
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green', 'orange']
    line_styles = ['-', '--', '-.', ':']
    for i, (alpha_val, rewards) in enumerate(alpha_curves.items()):
        # Apply rolling mean for smoother curves
        rewards_smooth = pd.Series(rewards).rolling(window=rolling_window).mean()
        plt.plot(rewards_smooth, color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)], 
                label=f'α = {alpha_val}', linewidth=2.5)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Learning Rate (α) Parameter Study', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot gamma curves
    plt.subplot(1, 3, 2)
    for i, (gamma_val, rewards) in enumerate(gamma_curves.items()):
        rewards_smooth = pd.Series(rewards).rolling(window=rolling_window).mean()
        plt.plot(rewards_smooth, color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)], 
                label=f'γ = {gamma_val}', linewidth=2.5)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Discount Factor (γ) Parameter Study', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot epsilon curves
    plt.subplot(1, 3, 3)
    for i, (epsilon_val, rewards) in enumerate(epsilon_curves.items()):
        rewards_smooth = pd.Series(rewards).rolling(window=rolling_window).mean()
        plt.plot(rewards_smooth, color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)], 
                label=f'ε = {epsilon_val}', linewidth=2.5)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Exploration Rate (ε) Parameter Study', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./Results/parameter_study_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_final_performance_comparison(alpha_curves, gamma_curves, epsilon_curves):
    """Plot final performance comparison"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Alpha final performance
    alphas = list(alpha_curves.keys())
    alpha_final = [np.mean(curve[-500:]) for curve in alpha_curves.values()]  # Average over last 500 episodes
    alpha_std = [np.std(curve[-500:]) for curve in alpha_curves.values()]
    
    bars1 = axes[0].bar(range(len(alphas)), alpha_final, yerr=alpha_std, 
                       color='skyblue', edgecolor='navy', capsize=5)
    axes[0].set_xlabel('Learning Rate (α)', fontsize=12)
    axes[0].set_ylabel('Average Final Reward', fontsize=12)
    axes[0].set_title('Final Performance: Learning Rate', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(alphas)))
    axes[0].set_xticklabels([f'{a}' for a in alphas])
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, alpha_final):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gamma final performance
    gammas = list(gamma_curves.keys())
    gamma_final = [np.mean(curve[-500:]) for curve in gamma_curves.values()]
    gamma_std = [np.std(curve[-500:]) for curve in gamma_curves.values()]
    
    bars2 = axes[1].bar(range(len(gammas)), gamma_final, yerr=gamma_std, 
                       color='lightgreen', edgecolor='darkgreen', capsize=5)
    axes[1].set_xlabel('Discount Factor (γ)', fontsize=12)
    axes[1].set_ylabel('Average Final Reward', fontsize=12)
    axes[1].set_title('Final Performance: Discount Factor', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(gammas)))
    axes[1].set_xticklabels([f'{g}' for g in gammas])
    axes[1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, gamma_final):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Epsilon final performance
    epsilons = list(epsilon_curves.keys())
    epsilon_final = [np.mean(curve[-500:]) for curve in epsilon_curves.values()]
    epsilon_std = [np.std(curve[-500:]) for curve in epsilon_curves.values()]
    
    bars3 = axes[2].bar(range(len(epsilons)), epsilon_final, yerr=epsilon_std, 
                       color='salmon', edgecolor='darkred', capsize=5)
    axes[2].set_xlabel('Exploration Rate (ε)', fontsize=12)
    axes[2].set_ylabel('Average Final Reward', fontsize=12)
    axes[2].set_title('Final Performance: Exploration Rate', fontsize=14, fontweight='bold')
    axes[2].set_xticks(range(len(epsilons)))
    axes[2].set_xticklabels([f'{e}' for e in epsilons])
    axes[2].grid(True, alpha=0.3)
    
    for bar, val in zip(bars3, epsilon_final):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./Results/parameter_study_final_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    best_alpha_idx = np.argmax(alpha_final)
    best_gamma_idx = np.argmax(gamma_final)
    best_epsilon_idx = np.argmax(epsilon_final)
    
    print("\n" + "="*80)
    print("DETAILED PARAMETER STUDY RESULTS")
    print("="*80)
    
    print("\nLearning Rate (α) Results:")
    for i, (alpha_val, final_val, std_val) in enumerate(zip(alphas, alpha_final, alpha_std)):
        marker = " *** BEST ***" if i == best_alpha_idx else ""
        print(f"  α = {alpha_val:4.2f}: {final_val:6.4f} ± {std_val:6.4f}{marker}")
    
    print("\nDiscount Factor (γ) Results:")
    for i, (gamma_val, final_val, std_val) in enumerate(zip(gammas, gamma_final, gamma_std)):
        marker = " *** BEST ***" if i == best_gamma_idx else ""
        print(f"  γ = {gamma_val:4.2f}: {final_val:6.4f} ± {std_val:6.4f}{marker}")
    
    print("\nExploration Rate (ε) Results:")
    for i, (epsilon_val, final_val, std_val) in enumerate(zip(epsilons, epsilon_final, epsilon_std)):
        marker = " *** BEST ***" if i == best_epsilon_idx else ""
        print(f"  ε = {epsilon_val:4.2f}: {final_val:6.4f} ± {std_val:6.4f}{marker}")
    
    print("\n" + "="*80)
    print("OPTIMAL PARAMETER COMBINATION:")
    print(f"Best α: {alphas[best_alpha_idx]} (Score: {alpha_final[best_alpha_idx]:.4f})")
    print(f"Best γ: {gammas[best_gamma_idx]} (Score: {gamma_final[best_gamma_idx]:.4f})")  
    print(f"Best ε: {epsilons[best_epsilon_idx]} (Score: {epsilon_final[best_epsilon_idx]:.4f})")
    print("="*80)

# Create Results directory if it doesn't exist
import os
os.makedirs('./Results', exist_ok=True)
