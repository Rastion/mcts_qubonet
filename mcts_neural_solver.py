import json
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from qubots.base_optimizer import BaseOptimizer

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # Current bitstring
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0
        self.cost = float('inf')

class QUBONet(nn.Module):
    """Dual-headed network for policy and value prediction"""
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1))
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh())
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.policy_head(features), self.value_head(features)

class MCTSNeuralOptimizer(BaseOptimizer):
    def __init__(self, num_simulations, exploration_c, rollout_depth, hidden_size, learning_rate, weight_decay, train_interval, verbose):
        super().__init__()
        self.num_simulations = num_simulations
        self.exploration_c = exploration_c
        self.rollout_depth = rollout_depth
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_interval = train_interval
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None

    def initialize_model(self, n):
        """Initialize network for problem size n"""
        self.model = QUBONet(n, self.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def ucb_score(self, node, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.value_sum / child.visit_count
            
        return q_value + self.exploration_c * child.prior * math.sqrt(node.visit_count) / (child.visit_count + 1)

    def select_child(self, node):
        return max(node.children, key=lambda child: self.ucb_score(node, child))

    def expand(self, node, qubo_matrix):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(node.state).to(self.device)
            policy, value = self.model(state_tensor)
        
        policy_probs = policy.cpu().numpy()
        selected_bits = []
        
        # Select bits using policy probabilities
        for bit in range(len(node.state)):
            if np.random.rand() < policy_probs[bit]:
                selected_bits.append(bit)
        
        # Ensure at least one child is generated
        if not selected_bits:
            selected_bits = [np.argmax(policy_probs)]
        
        # Create children
        children = []
        for bit in selected_bits:
            new_state = node.state.copy()
            new_state[bit] = 1 - new_state[bit]
            child = MCTSNode(new_state, parent=node)
            child.prior = policy_probs[bit]
            child.cost = self.calculate_cost(new_state, qubo_matrix)
            children.append(child)
        
        node.children.extend(children)
        return children[0]  # Return first child for simulation

    def simulate(self, node, qubo_matrix):
        current_state = node.state.copy()
        best_cost = node.cost
        for _ in range(self.rollout_depth):
            current_state = self.random_flip(current_state)
            current_cost = self.calculate_cost(current_state, qubo_matrix)
            if current_cost < best_cost:
                best_cost = current_cost
        return -best_cost  # Return negative cost as reward

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def random_flip(self, state):
        flip_idx = np.random.randint(len(state))
        new_state = state.copy()
        new_state[flip_idx] = 1 - new_state[flip_idx]
        return new_state

    def calculate_cost(self, state, qubo_matrix):
        return state @ qubo_matrix @ state.T

    def optimize(self, problem, initial_solution=None, **kwargs):
        QUBO_dict = problem.get_qubo()
        max_index = max(max(i, j) for i, j in QUBO_dict.keys())
        n = max_index + 1
        
        # Convert QUBO dict to matrix
        qubo_matrix = np.zeros((n, n))
        for (i, j), coeff in QUBO_dict.items():
            qubo_matrix[i][j] = coeff
            
                
        # Initialize model and root
        self.initialize_model(n)
        root_state = initial_solution if initial_solution is not None else np.random.randint(0, 2, n)
        root = MCTSNode(root_state)
        root.cost = self.calculate_cost(root_state, qubo_matrix)
        
        best_solution = root_state.copy()
        best_cost = root.cost
        print(f"RR {root.cost}")
        for sim in range(self.num_simulations):
                
            node = root
            # Selection
            while node.children:
                node = self.select_child(node)
                
            # Expansion
            if node.visit_count > 0 or not node.children:
                node = self.expand(node, qubo_matrix)
                if node is None:
                    break
                    
            # Simulation
            value = self.simulate(node, qubo_matrix)
            
            # Backpropagation
            self.backpropagate(node, value)
            print(node.cost)
            # Update best solution
            if node.cost < best_cost:
                best_cost = node.cost
                best_solution = node.state
                
            # Train network
            if sim % self.train_interval == 0:
                self.train_network(root, qubo_matrix)
                
        return best_solution.tolist(), float(best_cost)

    def train_network(self, root, qubo_matrix):
        states = []
        policies = []
        values = []
        
        # Collect training data
        queue = [root]
        while queue:
            node = queue.pop(0)
            states.append(node.state)
            visit_counts = [c.visit_count for c in node.children]
            total_visits = sum(visit_counts)
            policy = np.zeros(len(node.state))
            for c in node.children:
                flip_idx = np.where(c.state != node.state)[0][0]
                policy[flip_idx] = c.visit_count / total_visits
            policies.append(policy)
            values.append(-node.cost)  # Negative because lower cost is better
            queue.extend(node.children)
            
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        policies_tensor = torch.FloatTensor(np.array(policies)).to(self.device)
        values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1).to(self.device)
        
        # Training step
        self.model.train()
        pred_policies, pred_values = self.model(states_tensor)
        policy_loss = -torch.sum(policies_tensor * torch.log(pred_policies + 1e-10)) 
        value_loss = torch.nn.functional.mse_loss(pred_values, values_tensor)
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()