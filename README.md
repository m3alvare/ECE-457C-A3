# Assignment 3

Assignment description for course ECE 457C at the University of Waterloo in Spring 2024.

**Due Date:** July 30, 2024 by 11:59pm: submitted as PDF to the LEARN group dropbox for asg3.

**Collaboration:** You can discuss solutions and help to work out the code. The assignment can be done alone or as a pair, pairs do not need to be the same as for assignment 2. All code and writing will be cross-checked against each other and against internet databases for cheating. 

- Updates to code which will be useful for all or bugs found in the provided code will be updated on gitlab and announced on piazza.
- Be sure to try loading up the `Pytorch`, `Gynmasium` and `StableBaselines` frameworks as soon as possible to get through installation and library issues right at the start. If you leave it until the end and can't get things installed you're going to have a lot of unnecessary stress.
- Setting up some of these libraries and using the `Gymnasium` API, can be tricky, so *help each other on piazza* and we will monitor and try to improve the whole system for everyone. I'd rather everyone get past these library and installation issues as quickly as possible so they can focus on programming and training of the RL agents themselves.

## Resources

**OpenAI** has always had a great set of documentation (see ["Spinning Up in Deep RL"](https://spinningup.openai.com/en/latest/index.html)) of RL algorithms and standardized code, tools and experimental methods. 

Their "Taxonomy" list and papers are also very useful for this point in the course: https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#a-taxonomy-of-rl-algorithms

We'll be looking at a number of the papers they list in class, and for Asg 3, you need to read one of those papers and summarize parts you recognize from our discussions and think about how you could use the ideas in implementations.

`Gymnasium` is the successor of Open AI's original `Gym` project, which defines a standard set of environments for RL including an API for interacting with them.

`StableBaselines3` is a project to maintain a standard repository of core RL algorithms, and even trained models/policies. It uses the API defined in Gymnasium for interacting with RL environments, but SB3 is about policies, values functions, optimization, neural networks, gradients, etc., it isn't about environments themselves.



### Gitlab Repository for this assignment

- To share useful possible code

- https://git.uwaterloo.ca/ece457c/asg3-s24

- Example implementation for REINFORCE  which we'll talk about in class next week

- Example DQN code for the tutorial on July 8, 2024

  

## Domains for this Assignment

In the assignment you will use a few domains on Gymnasium to test out different Deep RL algorithms. 

1. `Lunar Lander` - with discrete actions - https://gymnasium.farama.org/environments/box2d/lunar_lander/
2. `Humanoid-v4` - continuous actions - https://gymnasium.farama.org/environments/mujoco/humanoid/
3. Another environment of your choice (we can talk about options)

## Assignment Requirements

This assignment will have a written component and a programming component. Your task is to train existing, or code your own and train, some Deep RL algorithms on these domains. There are two options for how to do this, using DeepRL libraries or coding it up yourself. 



- OPTION 1:
  - Train some algorithms provided by the [stable baselines](https://github.com/hill-a/stable-baselines) project and do some tuning of hyper-parameters for them. In the report, highlight the steps you tried to find the best hyper-parameter for all the algorithms. The same three algorithms and tuning approaches can be used for each environment. 
    - Stable Baselines RL Zoo : https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html
  - **(20%)** : Train and tune ***three*** RL algorithms from the stable baselines package and test on environment 1
  - **(20%)** : Train and tune ***three*** RL algorithms from the stable baselines package and test on environment 2
  - **(10%)** : Train and tune ***three*** RL algorithms from the stable baselines package and test on environment 3, these could be the same algorithms you've used on a different environment already.
- OPTION 2: 
  - **(50%)** Implement ***two*** RL algorithms from scratch (or any other algorithm, other than REINFORCE or DQN) from scratch using your own defined Deep Neural Networks and test on one of the environments. One algorithm should be *discrete action space* and the other should be *continuous action space*. The environment 1 and 2 above should be used to test the algorithm, but other environments can be used as well.
    - One approach could be by modifying the REINFORCE tutorial code presented in class
    - Another approach could be modifying the base stable-baselines algorithms for a simpler algorithm (for example: implement Prioritized Experience Replay for DB3 DQN, or some variant of PPO that isn't in SB3)
    - grading will be based on : design of networks; correct definitions of value functions, rewards, gradients, etc; code runs on both environments; performance is reasonably good compared to the baselines version (but it does *not* need to be equivalent to it)
- FOR BOTH OPTIONS: 
  - **(50%)** Report : Write a *short* report on the problem and the results of your algorithms. The report should be submited on crowdmark as a pdf. 
    - Define the MDP for each domain being used including the states, actions, dynamics and rewards. 
    - Define the mathematical formulation of each algorithm you are using, show the Bellman updates for you use. 
    - Some quantitative analysis of the results, a default plot for comparing all algorithms is given. You can do more than that based on types of analysis we've seen in class.
    - Clearly mention the hyper-parameters used and the steps that you took to arrive at the values you ended up using. 
    - Some qualitative analysis of why one algorithm works well in each case, and what you noticed along the way.
    - Note: if it is more convenient, you can report all of the results for one environment first, then all of the results for the second environment.

### Evaluation

You will also submit your code to LEARN and grading will be carried out by reading your code and comparing it to your report descriptions and results. We may run your code if needed to confirm the comparison. We will look at your definition and implementation which should match the description in the document.



## Installation

See the installation instructions for SB3 (https://github.com/DLR-RM/rl-baselines3-zoo) and their documentation on usage for various RL algorithms.

You should use Tensorboard to Monitor Training Progress so you can see if the system is going off track or not without needing to watch the environment itself.

[SB3 has other options for monitoring results](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/integrations.html), such as "Weights and Biases", which I've been meaning to try out, so you can look at those too.
