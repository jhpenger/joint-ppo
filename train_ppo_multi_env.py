from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonic_on_ray
import ray
import gym
from ray.rllib.agents import ppo
from ray.tune.registry import register_env


#env_name = 'sonic_env'
# Note that the hyperparameters have been tuned for sonic, which can be used
# run by replacing the below function with:
#
#     register_env(env_name, lambda config: sonic_on_ray.make(
#                                game='SonicTheHedgehog-Genesis',
#                                state='GreenHillZone.Act1'))
#
# However, to try Sonic, you have to obtain the ROM yourself (see then
# instructions at https://github.com/openai/retro/blob/master/README.md).
# register_env(env_name,
#              lambda config: sonic_on_ray.make(game='Airstriker-Genesis',
#                                               state='Level1'))


env_name = "multienv"


def assign_env(env_config):
    #print(env_config)
    print("worker index is {}".format(env_config.worker_index))
    if env_config.worker_index % 2 == 0:
        print('in even env')
        env = sonic_on_ray.make(game='Airstriker-Genesis', state='Level1')
    else:
        print('in odd env')
        env = sonic_on_ray.make(game='Airstriker-Genesis', state='Level1')

    return env

register_env(env_name, lambda env_config: assign_env(env_config))

ray.init()

config = ppo.DEFAULT_CONFIG.copy()

config.update({
    #"env_config": env_config,
    #'timesteps_per_batch': 40000,
    #'timesteps_per_batch': 8,
    "model": {
        "conv_filters": [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [256, [11, 11], 1],
        ],
    },
    "num_workers": 1,
    "train_batch_size": 2000,
    "sample_batch_size": 100,
    "lambda": 0.95,
    "clip_param": 0.2,
    "num_sgd_iter": 20,
    "lr": 0.0001,
    "sgd_minibatch_size": 32,
    "num_gpus": 1,
    'use_gae': True,
    'horizon': 4096,
    'kl_coeff': 0.0,
    'vf_loss_coeff': 0.0,
    'entropy_coeff': 0.0,

    # 'devices': ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5',
    #             '/gpu:6', 'gpu:7'],
    #'devices': ['/gpu:0'],
    #'num_gpus_per_worker':1,
    'tf_session_args': {
        'gpu_options': {'allow_growth': True}
    }
})

alg = ppo.PPOAgent(config=config, env=env_name)

for i in range(1000):
    result = alg.train()
    print('result = {}'.format(result))

    if i % 10 == 0:
        checkpoint = alg.save()
        print('checkpoint saved at', checkpoint)
