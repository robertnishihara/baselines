from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import ray
import tensorflow as tf, numpy as np
import time
# from baselines.common.mpi_adam import MpiAdam
# from baselines.common.mpi_moments import mpi_moments
# # from mpi4py import MPI
from collections import deque

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

# def learn(env, policy_fn, *,
#         timesteps_per_actorbatch, # timesteps per actor per update
#         clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
#         optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
#         gamma, lam, # advantage estimation
#         max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
#         callback=None, # you can do anything in the callback, since it takes locals(), globals()
#         adam_epsilon=1e-5,
#         schedule='constant' # annealing for stepsize parameters (epsilon and adam)
#         ):

def construct_policy(env, policy_fn, clip_param, entcoeff):
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    # adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    return pi, loss_names, lossandgrad, assign_old_eq_new, var_list


@ray.remote
class PPOActor(object):
    def __init__(self, env_creator, policy_fn,
            timesteps_per_actorbatch=None, # timesteps per actor per update
            clip_param=None, entcoeff=None, # clipping parameter epsilon, entropy coeff
            optim_epochs=None, optim_stepsize=None, optim_batchsize=None,# optimization hypers
            gamma=None, lam=None, # advantage estimation
            max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
            callback=None, # you can do anything in the callback, since it takes locals(), globals()
            adam_epsilon=1e-5,
            schedule='constant' # annealing for stepsize parameters (epsilon and adam)
            ):

        U.make_session(num_cpu=1).__enter__()

        self.env = env_creator()
        self.clip_param = clip_param
        self.entcoeff = entcoeff

        (self.pi,
         self.loss_names,
         self.lossandgrad,
         self.assign_old_eq_new,
         self.var_list) = construct_policy(self.env,
                                           policy_fn,
                                           self.clip_param,
                                           self.entcoeff)

        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.gamma = gamma
        self.lam = lam
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.max_iters = max_iters
        self.max_seconds = max_seconds
        self.callback = callback

        if schedule == 'constant':
            self.cur_lrmult = 1.0
        elif schedule == 'linear':
            self.cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        U.initialize()

        self.setfromflat = U.SetFromFlat(self.var_list)

        # adam.sync()

        # Prepare for rollouts
        # ----------------------------------------
        self.seg_gen = traj_segment_generator(self.pi, self.env, self.timesteps_per_actorbatch, stochastic=True)

        episodes_so_far = 0
        timesteps_so_far = 0
        self.iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

        assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

        # while True:
        #     if callback: callback(locals(), globals())
        #     if max_timesteps and timesteps_so_far >= max_timesteps:
        #         break
        #     elif max_episodes and episodes_so_far >= max_episodes:
        #         break
        #     elif max_iters and iters_so_far >= max_iters:
        #         break
        #     elif max_seconds and time.time() - tstart >= max_seconds:
        #         break
        #
        #     if schedule == 'constant':
        #         cur_lrmult = 1.0
        #     elif schedule == 'linear':
        #         cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        #     else:
        #         raise NotImplementedError

    def rollout(self):
        logger.log("********** Iteration %i ************"%self.iters_so_far)

        seg = self.seg_gen.__next__()
        add_vtarg_and_adv(seg, self.gamma, self.lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        self.d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not self.pi.recurrent)

        self.optim_batchsize = self.optim_batchsize or ob.shape[0]

        if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(ob) # update running mean/std for policy

        self.assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, self.loss_names))

    def reset_batches(self):
        self.batches = self.d.iterate_once(self.optim_batchsize)

    def get_gradients(self):
        batch = self.batches.__next__()
        *newlosses, g = self.lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], self.cur_lrmult)
        return g

    def set_params(self, params):
        self.setfromflat(params)


def main():

    ray.init()

    def env_creator():
        # from baselines.common.cmd_util import make_mujoco_env
        # env_id = "Humanoid-v1"
        # seed = 0
        # return make_mujoco_env(env_id, seed)
        import gym
        return gym.make("CartPole-v0")
        # return gym.make("Humanoid-v2")

    def policy_fn(name, ob_space, ac_space):
        from baselines.ppo1 import mlp_policy
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    schedule = 'constant'
    if schedule == 'constant':
        cur_lrmult = 1.0
    elif schedule == 'linear':
        cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
    else:
        raise NotImplementedError

    clip_param = 1
    timesteps_per_actorbatch = 10
    optim_batchsize = 5
    optim_stepsize = 0.01
    entcoeff = 0.5

    adam_epsilon = 1e-5
    actors = [PPOActor.remote(env_creator,
                              policy_fn,
                              timesteps_per_actorbatch=timesteps_per_actorbatch,
                              clip_param=clip_param,
                              entcoeff=entcoeff,
                              optim_epochs=10,
                              optim_stepsize=optim_stepsize,
                              optim_batchsize=optim_batchsize,
                              gamma=0.5,
                              lam=0.5,
                              max_timesteps=10) for _ in range(3)]

    U.make_session(num_cpu=1).__enter__()

    env = env_creator()
    clip_param = clip_param
    entcoeff = entcoeff

    (pi,
     loss_names,
     lossandgrad,
     assign_old_eq_new,
     var_list) = construct_policy(env,
                                  policy_fn,
                                  clip_param,
                                  entcoeff)

    U.initialize()

    adam = AdamOptimizer(var_list, epsilon=adam_epsilon)

    for i in range(100):
        print(i)
        [actor.rollout.remote() for actor in actors]

        optim_epochs = 10
        for _ in range(optim_epochs):

            [actor.reset_batches.remote() for actor in actors]

            for _ in range(timesteps_per_actorbatch // optim_batchsize):  # CHECK THAT THIS IS THE RIGHT NUMBER OF ITERATIONS!!!
                gradients = ray.get([actor.get_gradients.remote() for actor in actors])
                adam.update(gradients, optim_stepsize * cur_lrmult)
                params_id = ray.put(adam.getflat())
                [actor.set_params.remote(params_id) for actor in actors]


            # # Here we do a bunch of optimization epochs over the data
            # for _ in range(optim_epochs):
            #     losses = [] # list of tuples, each of which gives the loss for a minibatch
            #     for batch in d.iterate_once(optim_batchsize):
            #         *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            #         adam.update(g, optim_stepsize * cur_lrmult)
            #         losses.append(newlosses)
            #     logger.log(fmt_row(13, np.mean(losses, axis=0)))

            # logger.log("Evaluating losses...")
            # losses = []
            # for batch in d.iterate_once(optim_batchsize):
            #     newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            #     losses.append(newlosses)
            # meanlosses,_,_ = mpi_moments(losses, axis=0)
            # logger.log(fmt_row(13, meanlosses))
            # for (lossval, name) in zipsame(meanlosses, loss_names):
            #     logger.record_tabular("loss_"+name, lossval)
            # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            # lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            # listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            # lens, rews = map(flatten_lists, zip(*listoflrpairs))
            # lenbuffer.extend(lens)
            # rewbuffer.extend(rews)
            # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            # logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            # logger.record_tabular("EpThisIter", len(lens))
            # episodes_so_far += len(lens)
            # timesteps_so_far += sum(lens)
            # iters_so_far += 1
            # logger.record_tabular("EpisodesSoFar", episodes_so_far)
            # logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            # logger.record_tabular("TimeElapsed", time.time() - tstart)
            # if MPI.COMM_WORLD.Get_rank()==0:
            #     logger.dump_tabular()


class AdamOptimizer(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        # self.comm = MPI.COMM_WORLD if comm is None else comm

    # def update(self, localg, stepsize):
    def update(self, gradients, stepsize):
        # if self.t % 100 == 0:
        #     self.check_synced()
        gradients = [grad.astype('float32') for grad in gradients]
        # localg = localg.astype('float32')
        globalg = np.zeros_like(gradients[0])
        # self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        globalg = np.sum(gradients, axis=0)
        if self.scale_grad_by_procs:
            # globalg /= self.comm.Get_size()
            globalg /= len(gradients)

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    # def sync(self):
    #     theta = self.getflat()
    #     self.comm.Bcast(theta, root=0)
    #     self.setfromflat(theta)

    # def check_synced(self):
    #     if self.comm.Get_rank() == 0: # this is root
    #         theta = self.getflat()
    #         self.comm.Bcast(theta, root=0)
    #     else:
    #         thetalocal = self.getflat()
    #         thetaroot = np.empty_like(thetalocal)
    #         self.comm.Bcast(thetaroot, root=0)
    #         assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


if __name__ == "__main__":
    main()
