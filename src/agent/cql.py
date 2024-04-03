import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image



class CQL(object):
    def __init__(self, model, device, action_shape, args):
        self.model = model
        self.device = device
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.critic_encoder_tau
        self.image_size = args.agent_image_size
        self.log_interval = args.log_interval
        self.discount = args.discount
        self.detach_encoder = args.detach_encoder
        
        self.log_alpha = torch.tensor(np.log(args.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)
        #my additions
        self.batch_size = 128 #make adjustable
        self.min_q_version = 3
        self.min_q_weight = 10
        self.temp = 1.0
        self.image_size = 84
        self.lagrange_thresh = 0.0
        self.with_lagrange = False #turned off
        self.num_actions = 8
        self.cql_actor_lr = 0.0001
        self.cql_critic_lr = 0.0003
        self.alpha_prime_lr = 0.0

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.cql_actor_lr, betas=(args.actor_beta, 0.999))

        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), lr=self.cql_critic_lr, betas=(args.critic_beta, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))
        
        
        if self.with_lagrange:
            self.target_action_gap = self.lagrange_thresh
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device='cuda')
            #new optimizer for alpha_prime
            self.alpha_prime_optimizer = torch.optim.Adam(
                [self.log_alpha_prime], lr=self.alpha_prime_lr, betas=(args.critic_beta, 0.999))


        self.train()
        self.model.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.model.actor.train(training)
        self.model.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)
            
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.model.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def crop_observations(self, obs):
        obs_crop = torch.zeros(self.batch_size,9,84,84)
        for i in range(self.batch_size):
            obs_crop[i] = center_crop_image(obs[i], self.image_size) #make this importable or move out of function
        return obs_crop
    """
    def get_policy_actions_log_pi_slow(self, obs, num_actions):
        '''
        input: observation tensor, number_actions per observation
        output: actions, the agent would choose for current obersvations
        '''
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1, 1, 1).view(obs.shape[0] * num_actions, obs.shape[1], obs.shape[2], obs.shape[3])
        obs_temp_numpy = obs_temp.cpu().numpy()
        new_actions_numpy = np.zeros((self.batch_size*num_actions,6))
        new_log_pi = np.zeros((self.batch_size*num_actions,1))
        for i in range(self.batch_size*num_actions):
            new_actions_numpy[i], new_log_pi[i] = self.select_action_log_pi(obs_temp_numpy[i])
            
        current_actions_tensor = torch.from_numpy(new_actions_numpy).float()
        current_log_pi_tensor = torch.from_numpy(new_log_pi).float().view(obs.shape[0], num_actions, 1)
        combined = (current_actions_tensor, current_log_pi_tensor)
        return combined
    """
    #use numpy arrays or tensors throughout?
    #reparameterize?
    #without transfer cpu cuda
    def get_policy_actions_log_pi(self, obs, num_actions):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1, 1, 1).view(obs.shape[0] * num_actions, obs.shape[1], obs.shape[2], obs.shape[3])
        actions, _, log_pis, _ = self.model.actor(obs_temp, compute_log_pi =True)
        return actions, log_pis.view(obs.shape[0], num_actions, 1)

    def get_tensor_values(self, obs, actions, num_actions):
        """
        input: observations and actions
        output: q function estimate
        """
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape/obs_shape)
        
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1, 1, 1).view(obs.shape[0] * num_actions, obs.shape[1], obs.shape[2], obs.shape[3])
        (q1_pred, q2_pred) = self.model.critic(obs_temp,actions) #removed cuda()
        q1_pred = q1_pred.view(self.batch_size, num_repeat, 1)
        q2_pred = q2_pred.view(self.batch_size, num_repeat, 1)
        
        return (q1_pred, q2_pred)

    #try with log_pi
    def select_action_log_pi(self, obs):
        if obs.shape[-1] != self.image_size:
            print('cropping_occured_1')
            obs = center_crop_image(obs, self.image_size)
            
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, log_pi, _ = self.model.actor(
                obs, compute_pi=True, compute_log_pi=True
            )
            (mu,log_pi) = mu.cpu().data.numpy().flatten(), log_pi.cpu().data.numpy().flatten()
            return (mu, log_pi)

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.model.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()
    

    def get_conservative_loss(self, obs, action, reward, next_obs, not_done, L, step, num_actions):
        
        #crop observations
        #cropped_observations = obs #self.crop_observations(obs)
        #new_cropped_observations =  next_obs #self.crop_observations(next_obs)
        print('dimensions_in conservative loss after second cropping:')
        print(obs.shape)
        #get random actions tensor
        random_actions_tensor = torch.cuda.FloatTensor(obs.shape[0]*num_actions,action.shape[-1]).uniform_(-1,1) #.to(device)
        
        #use policy to get actions and log_pis for current and next observations
        curr_actions_tensor, curr_log_pis = self.get_policy_actions_log_pi(obs, num_actions)
        new_curr_actions_tensor, new_log_pis = self.get_policy_actions_log_pi(next_obs, num_actions)


        #log action_diff for batch
        #action_diff = (((action-curr_actions_tensor)**2) **0.5)
        #L.log('train/action_diff', action_diff,step)

        #Predict Q Values for these (observation,action)-tensors
        q1_pred, q2_pred = self.model.critic(obs, action) #(self.crop_observations(obs).cuda(), action) # remove crop
        q1_curr_actions, q2_curr_actions = self.get_tensor_values(obs, curr_actions_tensor, num_actions)
        q1_next_actions, q2_next_actions = self.get_tensor_values(obs, new_curr_actions_tensor, num_actions)
        q1_rand, q2_rand = self.get_tensor_values(obs, random_actions_tensor, num_actions)

        #q1_pred has shape [batch_size, 1, 1]
        #all others have the shape [batch_size, num_actions, 1]

        #put q-values in one vector
        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions],1 
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions],1 
        )
       
        if self.min_q_version == 3:
            #add importance sampling for CQL(H)
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )  
        
        #calculate conservative loss
        min_qf1_loss = torch.logsumexp(cat_q1/self.temp, dim=1).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2/self.temp, dim=1).mean() * self.min_q_weight * self.temp
        #subtract the log likelihood of the data
        
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

        #calculate metrics and log them
        metrics = {}
        metrics['q1_std'] = torch.std(cat_q1, dim = 1).mean()
        metrics['q2_std'] = torch.std(cat_q2, dim = 1).mean()
        metrics['q1_current_actions'] = q1_curr_actions.mean()
        metrics['q2_current_actions'] = q2_curr_actions.mean()
        metrics['q1_rand'] = q1_rand.mean()
        metrics['q2_rand'] = q2_rand.mean()
        metrics['q1_actions'] = q1_pred.mean()
        metrics['q2_actions'] = q2_pred.mean()

        for keys, contents in metrics.items():
            L.log('train/'+keys, contents, step)

        return (min_qf1_loss, min_qf2_loss) #changed + to ,
                   


    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.model.actor(next_obs)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(
            obs, action, detach=self.detach_encoder)
        
        
        bellman_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        #add conservative loss
        min_qf1_loss, min_qf2_loss = self.get_conservative_loss(obs, action, reward, next_obs, not_done, L, step, self.num_actions)
        
        if step % self.log_interval == 0:
            L.log('train_critic/conservative_loss', min_qf1_loss+min_qf2_loss, step)
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0 )
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        critic_loss = bellman_loss + min_qf1_loss + min_qf2_loss

        if step % self.log_interval == 0:
            L.log('train_critic/ciritic_loss', critic_loss, step)
            L.log('train_critic/conservative_loss_adjusted', min_qf1_loss+min_qf2_loss, step)
            L.log('train_critic/bellman_error', bellman_loss, step)
            #L.log('train_critic/alpha_prime_value', alpha_prime, step)
            #L.log('train_critic/alpha_prime_loss', alpha_prime_loss, step)
            #L.log('train_critic/std_q1', std_q1, step)
            #L.log('train_critic/std_q2', std_q2, step)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.model.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.model.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.model.soft_update_params(self.critic_tau, self.encoder_tau)


    ###Anpassungen
    #--> replay_buffer returned das äquivalent zum batch
    #aus meinem repo: sample oder select action?
   

   
    ###add CQL by Adrian

    ##tensors:
    #actions.shape and batch size?
    #random actions tensor qf1 has output size = 1 --> random_actions_tensor.shape torch.Size([10, 6])
    #current actions tensor, current logpis--> get 10 actions and logpis from current policy for current observation
    #new actions tensor, new logpis--> get 10 actions and logpis from current policy for next observation
    
    ##q1 and q2 predictions of the above tensors?
    # --> need to implement a function like this too?
    #q1_rand
    #q2_rand
    #q1_curr_actions
    #q2_curr_actions
    #q1_next_actions
    #q2_next_actions




    def save_model(self, dir, step):
        torch.save(self.model.state_dict(), os.path.join(dir, f'{step}.pt'))

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))  