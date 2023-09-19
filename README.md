# POLICY EVALUATION

## AIM:
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT:
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
The environment has 7 states:

* Two Terminal States: G: The goal state & H: A hole state.
* Five Transition states / Non-terminal States including S: The starting state.

### Actions
The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities
The transition probabilities for each action are as follows:

* 50% chance that the agent moves in the intended direction.
* 33.33% chance that the agent stays in its current state.
* 16.66% chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/ee9c6dcf-b579-4b1c-9663-47c8b17a08b4)

## POLICY EVALUATION FUNCTION

### Formula
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/574fb688-7c9f-409f-b07f-e75441d8f4b3)

## Policy Evaluation:
```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
         V = np. zeros (len (P))
         for s in range (len (P)):
           for prob, next_state, reward, done in P[s] [pi (s)]:
            V[s] +=prob * (reward + gamma *  prev_V[next_state] * (not done))
         if np.max(np.abs (prev_V - V)) < theta:
            break
         prev_V =V.copy()
    return V
# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)
# Code to evaluate the second policy
# Write your code here
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)
# Comparing the two policies
# Compare the two policies based on the value function using the above equation and find the best policy
V1
print_state_value_function(V1, P, n_cols=7, prec=5)
V2
print_state_value_function(V2, P, n_cols=7, prec=5)
V1>=V2
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")

```

## OUTPUT:
### Policy 1:
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/8569d152-1cbc-42b2-af2c-1abe3dc34209)
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/c3227142-e859-433c-ae52-b0221f22b639)
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/3cb8176a-d6c7-42b3-a13b-8fa6eea829e5)



### Policy 2:
![image](https://github.com/sangeethak15-AI/rl-policy-evaluation/assets/93992063/0d254e9e-4aab-4126-9cb9-935285b5bd1a)
![image](https://github.com/sangeethak15-AI/rl-policy-evaluation/assets/93992063/b8a640a3-25dc-4a9e-808f-5dc22c6e664b)
![image](https://github.com/sangeethak15-AI/rl-policy-evaluation/assets/93992063/f12a98f1-d26e-41dd-a4c9-f92dca77789d)


### Comparison:
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/1ee6d217-5ada-4fc0-ae3d-e3701e13746a)


### Conclusion:
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/18f14f65-c243-4b57-b72d-4f077e0cbe96)


## RESULT:
Thus, a Python program is developed to evaluate the given policy.
