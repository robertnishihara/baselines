#!/usr/bin/env python3
import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2_ray
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import multiprocessing
import ray
import tensorflow as tf


def train(env_id, num_timesteps, seed, policy, num_workers):

    # ncpu = multiprocessing.cpu_count()
    # if sys.platform == 'darwin': ncpu //= 2
    # config = tf.ConfigProto(allow_soft_placement=True,
    #                         intra_op_parallelism_threads=ncpu,
    #                         inter_op_parallelism_threads=ncpu)
    # config.gpu_options.allow_growth = True #pylint: disable=E1101
    # tf.Session(config=config).__enter__()

    def env_creator():
        env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
        return env

    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]
    ppo2_ray.learn(policy=policy, env_creator=env_creator, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        num_workers=num_workers)

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    args = parser.parse_args()
    logger.configure()

    num_workers = 3

    ray.init(num_workers=0)

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, num_workers=num_workers)

if __name__ == '__main__':
    main()
