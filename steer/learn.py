import numpy as np
import gym

from .agent import Agent, TDAgent

if __name__ == '__main__':

  # Hardcoded settings
  n_episodes = 100
  path       = "tdagent.bin"

  # Create an env, load or create an agent
  env   = gym.make('CarRacing-v0')
  agent = Agent.load(path, TDAgent(learning_rate=0.75, alpha=0.9))
  print("Agent knows {} policies".format(len(agent.policy)))

  # Preset of actions (stolen from Nawar's ideas)
  actions = [np.array(v) for v in [
    [-.90, 0, 0],
    [-.45, 0, 0],
    [0, 0, 0],
    [ .45, 0, 0],
    [ .90, 0, 0],
    [-.90, .5, 0],
    [ .90, .5, 0],
    [  0,  .5, 0],
    [ .45, .5, 0],
    [-.45, .5, 0]
  ]]

  # Start!
  print("Starting the learning episodes")
  best_reward = 0
  for i in range(n_episodes):
    
    observation = env.reset()
    print("Episode {} of {} ...".format(i+1, n_episodes))
    
    n = 0
    done = False
    last_action = -1
    last_state = None
    total_reward = 0

    last_reward = 0
    num_consecutive_reduction = 0

    while not done:
      n = n+1
      env.render()

      action,_ = agent.best_action(observation)

      # If the bot does not know how to react,
      # random from the action space
      if action == -1:
        # Take random action, blindly
        action = actions[np.random.choice(len(actions))]
      elif action is None:
        # Random action too
        action = env.action_space.sample()
      else:
        action = agent.encoder.decode_action(action)

      new_observation, reward, done, info = env.step(action)
      total_reward += reward

      if reward <= last_reward:
        num_consecutive_reduction += 1
      else:
        num_consecutive_reduction = 0

      last_reward = reward

      # Record best score
      if total_reward > best_reward:
        best_reward = total_reward

      # Learn
      agent.learn(observation, action, reward, new_observation)

      observation = new_observation

      if done or ((total_reward <= 0 or num_consecutive_reduction > 20) and n > 300):
        print("... Episode DONE!")
        print("... The agent knows {} observations so far".format(len(agent.policy)))
        agent.encoder.n = 0
        done = True
        # Save the trained agent
        agent.save(path)

    print("Best score so far : ", best_reward)

  
