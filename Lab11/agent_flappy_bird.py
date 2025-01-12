import copy
import flappy_bird_gym
import torch
from torch import nn
import random
import time
import cv2
import numpy as np
import os
import imageio
import re

if not os.path.exists("images"):
    os.makedirs("images")
os.makedirs('er', exist_ok=True)
filename = "values.txt"

env = flappy_bird_gym.make("FlappyBird-rgb-v0")

IN_FRAMES = 4
def processed_image(image, resize=(110, 84), crop=(180, 402), index=0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image[0:crop[0], 0:crop[1]]
    image = image[220 - 169:, :]

    image = cv2.resize(image, resize, interpolation=cv2.INTER_LINEAR)
    image = image / 255.

    #cv2.imwrite(f"images/image_{index}.png", (image * 255).astype(np.uint8))
    return image

def get_images(image, frames, index=0):
    processed_img = processed_image(image, index=index)

    if frames is None or len(frames) == 0:
        frames = [processed_img] * (IN_FRAMES - 1)

    frames = frames + [processed_img]
    stacked_frames = np.stack(frames)
    previous_frames = frames[-(IN_FRAMES - 1):]

    return stacked_frames, previous_frames

class DQN(nn.Module):
    def __init__(self, count_moves):
        super(DQN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(IN_FRAMES, 32, kernel_size=8, stride=4),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2),nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1),nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(64 * 7 * 10, 512),nn.ReLU()) # (size_in-kernel+2*padding)/stride+1
        self.layer5 = nn.Sequential(nn.Linear(512, count_moves))

    def forward(self, image):
        q_values = self.layer1(image)
        q_values = self.layer2(q_values)
        q_values = self.layer3(q_values)

        q_values = q_values.view(-1, 64 * 7 * 10)
        q_values = self.layer4(q_values)
        q_values = self.layer5(q_values)

        return q_values

    def train_minibatch(self, target_model, optimizer, image, count_moves, rewards, next_image, done, gamma=0.99):
        max_next_moves = torch.max(self.forward(next_image), dim=1)[1].detach() #0-val maxima, 1-poz

        target_next_q = target_model.forward(next_image)
        max_next_q = target_next_q.gather(index=max_next_moves.view(-1, 1), dim=1).view(-1).detach()

        actual_q = rewards + (1 - done) * gamma * max_next_q

        pred_q = self.forward(image)
        pred_q = pred_q.gather(index=count_moves.view(-1, 1), dim=1).view(-1)

        loss = torch.mean((actual_q - pred_q) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class ExperienceReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []

    def add_move(self, move_inf):
        self.data.append(move_inf)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]

    def moves_state(self, n):
        n = min(n, len(self.data))
        ids = np.random.choice(range(len(self.data)), n, replace=False)
        samples = [self.data[i] for i in ids]

        image = torch.tensor(np.stack([s[0] for s in samples])).float()
        move = torch.tensor([s[1] for s in samples]).long()
        reward = torch.tensor([s[2] for s in samples]).float()
        next_image = torch.tensor(np.stack([s[3] for s in samples])).float()
        done = torch.tensor([s[4] for s in samples]).float()
        return image, move, reward, next_image, done

count_moves = env.action_space.n
frame_skip = 3
train_batch_size = 64
update_freq = 4
episodes = 100001
experience_memory = 50000
learning_rate = 2.5e-4
save_interval = 1000
print_freq = 1000
target_update = 10000


def initialize_params(filename):
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r"Step:\s*(\d+),\s*Epsilon:\s*([\d.]+),\s*Episode:\s*(\d+),\s*Best_score:\s*(\d+)", line)
            if match:
                step = int(match.group(1))
                epsilon = float(match.group(2))
                episode = int(match.group(3))
                best_score = int(match.group(4))

    return step, epsilon, episode+1, best_score

def epsilon(step, epsilon_decay=1e6):
    value = 1 - 0.9 * (step / epsilon_decay)
    return max(0.0001, min(value, 1))

def training():
    global_step = 0
    max_episode = 0
    best_score = 0
    score_video = 0
    episode = 0
    er = ExperienceReplay(experience_memory)
    #global_step, _, episode, best_score = initialize_params("values.txt") ###
    #start_ep = episode                                                    ###
    #model = torch.load(f'models/model_{episode-1}.pt')                    ###
    model = DQN(count_moves=count_moves)                                   #
    #target_model = torch.load(f'models/model_{episode-1-target_update}.pt') ##
    #loaded_data = torch.load(f'er/er_checkpoint{episode-1}.pt')
    #er.data = loaded_data
    target_model = copy.deepcopy(model)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-6) #
    all_rewards = []
    all_average_rewards = []
    best_episode_frames = [] ##video

    while episode < episodes:
        episode_frames = []  ##video
        prev_frames = []
        image, prev_frames = get_images(env.reset(), prev_frames, global_step)

        episode_reward = 0
        while True:
            frame = env.render(mode='rgb_array')  ##video
            episode_frames.append(frame)          ##video

            if np.random.rand() < epsilon(global_step):
                act = 0 if random.random() > 0.1 else 1
            else:
                img_tensor = torch.tensor(image).float()
                q_values = model(img_tensor)[0]
                q_values = q_values.detach().numpy()
                act = np.argmax(q_values)

            next_image, reward, done, _ = env.step(act)
            global_step += 1

            episode_reward += reward

            if act and not done and random.random() > 0.1:
                skip_reward = 0
                for _ in range(frame_skip):
                    next_image, reward, done, _ = env.step(0)
                    global_step += 1
                    skip_reward += reward
                    if done :
                        frame = env.render(mode='rgb_array')  ##video
                        episode_frames.append(frame)          ##video
                        break

                    frame = env.render(mode='rgb_array')   ##video
                    episode_frames.append(frame)           ##video
                episode_reward += skip_reward

            next_image, prev_frames = get_images(next_image, prev_frames)

            if done:
                if episode_reward > best_score:
                    best_score = episode_reward
                    best_episode_frames = episode_frames   ##video
                if episode_reward > max_episode:
                    max_episode = episode_reward
                if episode_reward == 101:
                    reward = - 50
                elif episode_reward > 101:
                    reward = - 10
            else:
                if episode_reward > 101:
                    reward = reward + 1
                else:
                    reward = reward + 0.1
            er.add_move([image, act, reward, next_image, int(done)])
            image = next_image


            if global_step % update_freq == 0 or act:
                image_state, act_state, reward_state, next_image_state, done_state = er.moves_state(train_batch_size)
                model.train_minibatch(target_model, optimizer, image_state, act_state, reward_state, next_image_state,done_state)

            if global_step and global_step % target_update == 0:
                target_model = copy.deepcopy(model)

            if done:
                break

        all_rewards.append(episode_reward)

        if episode and episode % print_freq == 0:
            print(f"Max_episode: {max_episode}")
            max_episode = 0
            all_average_rewards.append(np.mean(all_rewards[-print_freq:]))
            print('Episode #{} | Step #{} | Epsilon {:.2f} | Average reward {:.2f}'.format(
                episode, global_step, epsilon(global_step), np.mean(all_rewards[-print_freq:])))

        if episode and episode % save_interval == 0:
            if score_video < best_score:
                if best_episode_frames:
                    score_video = best_score
                    with imageio.get_writer('best_episode.mp4', fps=30) as video:
                        for frame in best_episode_frames:
                            video.append_data(frame)

            torch.save(model, f'models/model_{episode}.pt')
            with open(filename, "w") as file:
                    eps_value = epsilon(global_step)
                    file.write(f"Step: {global_step}, Epsilon: {eps_value:.4f}, Episode: {episode}, Best_score: {best_score}\n")

        if episode and episode % 10000 == 0:
            torch.save(er.data, f'er/er_checkpoint{episode}.pt')

        episode += 1
#training()

def test():
    model = torch.load('models/model_10000.pt')
    step = 0
    best_score = -1

    for i in range(20):
        prev_frames = []
        total_reward = 0
        image, prev_frames = get_images(env.reset(), prev_frames, step)

        while True:
            img_tensor = torch.tensor(image).float()
            q_values = model(img_tensor)[0]
            q_values = q_values.detach().numpy()
            move = np.argmax(q_values)

            next_img, reward, done, _ = env.step(move)
            total_reward += reward

            if done:
                break
            env.render()
            time.sleep(1 / 300)

            image, prev_frames = get_images(next_img, prev_frames)

        if total_reward > best_score:
            best_score = total_reward

    print(f"Best score: {best_score}")
test()
