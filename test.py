from flappy_bird import FlappyBird

from dqn_agent import DQN_Agent


def train():
    env = FlappyBird()
    agent = DQN_Agent(action_size=2, train=False)

    image, reward, terminal = env.next_frame(action=0)
    while True:
        action = agent.act(image)
        print("action: ", action)
        next_image, reward, terminal = env.next_frame(action)
        image = next_image

        if terminal:
            env = FlappyBird()
            image, reward, terminal = env.next_frame(action=0)


if __name__ == '__main__':
    train()
