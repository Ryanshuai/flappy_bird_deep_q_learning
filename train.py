from flappy_bird import FlappyBird

from dqn_agent import DQN_Agent


def train():
    env = FlappyBird()
    agent = DQN_Agent(action_size=2)

    image, reward, terminal = env.next_frame(action=0)
    reward_sum = reward
    while True:
        action = agent.act(image)
        next_image, reward, terminal = env.next_frame(action)
        reward_sum += reward
        agent.observe(image, action, reward, next_image, terminal)
        agent.learn()

        image = next_image

        if terminal:
            agent.writer.add_scalar("reward_sum", reward_sum, agent.learn_counter)
            print("reward_sum: ", reward_sum)
            env = FlappyBird()
            image, reward, terminal = env.next_frame(action=0)
            reward_sum = reward


if __name__ == '__main__':
    train()
