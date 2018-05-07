# DeepQLearning
This is a project by team **Invictus** for a course on Advanced Machine Learning.
### **Overview**
We aimed to successfully learn control policies directly from high dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.
### **Related files**
* train.py
* testrun.py

### **Descriptions of the project files**
#### train.py
Implementation of DQN and it's training. 
#### testrun.py
Using the weights generated through training, the emulator runs through 10 episodes and reports the corresponding average score
#### checkpoint, model_breakout.ckpt.index, model_breakout.ckpt.meta, model_breakout.ckpt.data-00000-of-00001
Together they represent the saved state of the model
