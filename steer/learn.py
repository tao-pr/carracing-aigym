import numpy as np
import gym

from .agent import Agent, TDAgent

if __name__ == '__main__':

  # Hardcoded settings
  n_episodes = 20
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
  for i in range(n_episodes):
    
    observation = env.reset()
    print("Episode {} of {} ...".format(i+1, n_episodes))
    
    n = 0
    done = False
    last_action = -1
    last_state = None

    while not done:
      n = n+1
      env.render()

      # Given the current state, ask the agent to find the best action to take
      action,_ = agent.best_action(observation)

      # If the bot does not know how to react,
      # random from the action space
      if action == -1:
        # action = env.action_space.sample()
        # Take random action, blindly
        action = actions[np.random.choice(len(actions))]
      else:
        action = agent.encoder.decode_action(action)

      new_observation, reward, done, info = env.step(action)

      # Learn
      agent.learn(observation, action, reward, new_observation)

      if done:
        print("... Episode DONE!")
        print("... The agent knows {} observations so far".format(len(agent.policy)))
        agent.encoder.n = 0
        # TODO reset the agent for the next episode

  # Save the trained agent
  agent.save(path)
