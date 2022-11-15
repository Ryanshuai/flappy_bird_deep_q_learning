from flappy_bird import FlappyBird

from dqn_agent import DQN_Agent


def observe():
    env = FlappyBird()
    agent = DQN_Agent(action_size=2)
    agent.random_action_probability_decay = 1.0

    image, reward, terminal = env.next_frame(action=0)
    reward_sum = reward
    for i in range(50_000):
        action = agent.act(image)
        next_image, reward, terminal = env.next_frame(action)
        reward_sum += reward
        agent.observe_to_disk(image, action, reward, next_image, terminal)
        image = next_image

        if terminal:
            print("reward_sum: ", reward_sum)
            env = FlappyBird()
            image, reward, terminal = env.next_frame(action=0)
            reward_sum = reward


if __name__ == '__main__':
    observe()
