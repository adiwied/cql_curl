import sys
sys.path.append('.')

import numpy as np
import torch

from env import make_envs
from utils.argument import parse_args
from utils.misc import set_seed_everywhere, make_dir, VideoRecorder, eval_mode
from utils.logger import Logger
from memory import ReplayBufferStorage, make_replay_buffer
from model.model import CURL_Model
from agent import CURL
from pathlib import Path


import time
import os
import json
import wandb

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

torch.backends.cudnn.benchmark = True

def evaluate(env, agent, video, num_episodes, L, step, tag=None):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i==0))
        done = False
        episode_reward = 0
        while not done:
            with eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        if L is not None:
            video.save(f'{step}.mp4')
            #L.log(f'eval/episode_reward', episode_reward, step)
        episode_rewards.append(episode_reward)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if L is not None:
        L.log(f'eval/mean_reward', mean_reward, step)
        L.log(f'eval/std_reward', std_reward, step)
    return mean_reward

def main():

    args = parse_args()
    args.agent = 'curl'
    device = 'cuda'
    args.save_tb = True

    args.frame_stack = 3
    args.action_repeat = 2
    args.num_train_steps = 1000000
    args.eval_freq = 5000
    args.init_steps= 1000
    args.env_image_size = 100
    args.num_eval_episodes =10
    args.save_video = True
    args.save_model= True
    args.domain_name = 'walker'
    args.task_name = 'walk'
    args.seed = 2
    print(args)
    set_seed_everywhere(2)

    run = wandb.init(
            project="train_behavior_policy",
            entity="adrianw",
            name='sac_curl_online',
            config=args,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    wandb.tensorboard.patch(root_logdir=f'{args.work_dir}/tb', pytorch=True)

    ts = time.strftime("%m-%d", time.gmtime())
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.env_image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.agent + 'sac_curl_online'
    args.work_dir = args.work_dir + '/'  + exp_name
    make_dir(args.work_dir)
    video_dir = make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = make_dir(os.path.join(args.work_dir, 'model'))

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['MUJOCO_GL'] = 'egl'
    video = VideoRecorder(video_dir if args.save_video else None)

    print(f"Working Direktory: {args.work_dir}")

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # prepare env
    env = make_envs(args)
    eval_env = make_envs(args)

    action_shape = env.action_space.shape
    agent_obs_shape = (3*args.frame_stack, args.agent_image_size, args.agent_image_size)
    env_obs_shape = (3*args.frame_stack,args.env_image_size, args.env_image_size)

    action_shape = env.action_space.shape
    print(f"action_shape: {action_shape}")
    observation_shape = env.observation_space.shape
    print(f"observation_shape: Agent {agent_obs_shape}, Environment {env_obs_shape}")

    #Setting up model and agent
    curl_model = CURL_Model(obs_shape = agent_obs_shape,
                        action_shape        = action_shape,       
                        hidden_dim          = args.hidden_dim,
                        encoder_feature_dim = args.encoder_feature_dim,
                        log_std_min         = args.actor_log_std_min,
                        log_std_max         = args.actor_log_std_max,
                        num_layers          = args.num_layers, 
                        num_filters         = args.num_filters, 
                        device  = device)

    args.detach_encoder

    agent = CURL(model       = curl_model, 
                device      = device, 
                action_shape=action_shape,  
                args        = args)

    replay_buffer = None
    L = Logger(args.work_dir, use_tb=args.save_tb, config=args.agent)
    replay_storage = ReplayBufferStorage(Path(args.work_dir) / 'buffer')

    episode, episode_reward, done, info = 0, 0, True, {}
    start_time = time.time()

    for step in range(args.num_train_steps+1):
        if step%500==0:
            print(f"step: {step}")
        # evaluate agent periodically

        if step > args.init_steps and step % args.eval_freq == 0:
            print("evaluation")
            L.log('eval/episode', episode, step)
            with torch.no_grad():
                evaluate(eval_env, agent, video, 10, L, step)
            if args.save_model and step % (10*args.eval_freq) == 0:
                agent.save_model(model_dir, step)

        if done:
            if step > 0:
                replay_storage.add(obs, None, None, True)  # removed comment
                if step % args.log_interval == 0:
                    L.log('train/episode_reward', episode_reward, step)
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            print("episode", episode)
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            if replay_buffer is None:
                replay_buffer = make_replay_buffer(replay_dir=Path(args.work_dir) / 'buffer',
                                                max_size=args.replay_buffer_capacity,
                                                batch_size=args.batch_size,
                                                num_workers=1,
                                                save_snapshot=True,
                                                nstep=1,
                                                discount=args.discount,
                                                obs_shape=env_obs_shape,
                                                device=device,
                                                image_size=args.agent_image_size,
                                                image_pad=args.image_pad)
                print(replay_buffer.sample)


            num_updates = 1 if step > args.init_steps else args.init_steps
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, info = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        replay_storage.add(obs, action, reward, done_bool)    

        obs = next_obs
        episode_step += 1       

    if run != None:
        run.finish()

if __name__ == '__main__':
    main()
