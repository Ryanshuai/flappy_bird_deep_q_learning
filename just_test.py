from flappy_bird import FlappyBird
from dqn_agent import DQN_Agent


def train():
    agent = DQN_Agent(action_size=2, train=False, pth="pths/4_6.pth")

    env = FlappyBird()
    image, reward, terminal = env.next_frame(action=0)
    reward_sum = reward

    while True:
        action = agent.act(image)
        next_image, reward, terminal = env.next_frame(action)
        reward_sum += reward
        image = next_image

        if terminal:
            agent.writer.add_scalar("Reward/reward_sum", reward_sum, agent.learn_counter)
            print("reward_sum: ", reward_sum)
            env = FlappyBird()
            image, reward, terminal = env.next_frame(action=0)
            reward_sum = reward


if __name__ == '__main__':
    train()
