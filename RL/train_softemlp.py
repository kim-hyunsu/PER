from jax import jit, vmap
from jax_rl.evaluation import evaluate, rpp_evaluate, softemlp_evaluate
from emlp.groups import *
from RL.representations_deprecated import environment_symmetries
from jax_rl.utils import make_env
from jax_rl.datasets import ReplayBuffer
from jax_rl.agents import AWACLearner, SACLearner, SoftSACLearner
import os
import random

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import wandb

import sys
sys.path.append("../")

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Ant-v2', 'Environment name.')
flags.DEFINE_string('add_envname', '', 'Environment name.')
flags.DEFINE_boolean('logoff', False, 'No wandb')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('search_interval', 10000, 'Model search interval.')
flags.DEFINE_boolean('gnl', False, 'gated nonlinearity')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('rpp_value', False, 'Use RPP for value function')
flags.DEFINE_boolean('rpp_policy', True, 'Use RPP for policy function')
flags.DEFINE_string('group', '', 'Also use RPP for value function')
flags.DEFINE_float('equiv_wd', 1e-6, 'Policy Equivariant weight decay')
flags.DEFINE_float('basic_wd', .33, 'Policy Basic weight decay')
flags.DEFINE_string('equiv', '70', 'Policy Equivariant weight decay')
flags.DEFINE_string('wd', '0', 'Policy Basic weight decay')
flags.DEFINE_float('cequiv_wd', 1e-6, 'Critic Equivariant weight decay')
flags.DEFINE_float('cbasic_wd', .33, 'Critic Basic weight decay')
flags.DEFINE_string('cequiv', '70', 'Critic Equivariant weight decay')
flags.DEFINE_string('cwd', '0', 'Critic Basic weight decay')
flags.DEFINE_list('hidden_dims', [256, 256], 'Dimension of hidden layers')
flags.DEFINE_boolean('small_init', True,
                     'Use smaller init for last policy layer')
flags.DEFINE_boolean('old_rep', False, "Use original rep allocation heuristic")
flags.DEFINE_boolean("gan_betas", False, "use GAN betas or not")
flags.DEFINE_float("tau", 0.005, 'tau for SAC updates')
flags.DEFINE_boolean('standardize', False,
                     "Use equivariant standardization of the state")
flags.DEFINE_float('clipping', 0.5, 'Gradient Norm magnitude at which to clip')
flags.DEFINE_integer(
    'ncritic', 1, 'Number of critic updates per policy update')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    print("CWD = ", os.getcwd())
    fname = str(FLAGS.seed) + "equivWD" + str(FLAGS.equiv_wd) +\
        "_basicWD" + str(FLAGS.basic_wd) +\
        "_cequivWD" + str(FLAGS.cequiv_wd) +\
        "_cbasicWD" + str(FLAGS.cbasic_wd)
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, FLAGS.env_name, fname))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    kwargs.update(environment_symmetries[FLAGS.env_name])

    hidden_dims = tuple(int(hd) for hd in FLAGS.hidden_dims)
    kwargs['hidden_dims'] = hidden_dims

    kwargs['rpp_value'] = FLAGS.rpp_value
    kwargs['rpp_policy'] = FLAGS.rpp_policy
    if FLAGS.group:
        kwargs['symmetry_group'] = eval(FLAGS.group)
        kwargs['state_rep'] = kwargs['state_rep'](kwargs['symmetry_group'][0])
        kwargs['action_rep'] = kwargs['action_rep'](kwargs['symmetry_group'][0])
    if FLAGS.old_rep:
        kwargs.pop('middle_rep', None)

    replay_buffer_size = kwargs.pop('replay_buffer_size')
    action_dim = env.action_space.shape[0] if kwargs['action_space'] == 'continuous' else 1
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 replay_buffer_size or FLAGS.max_steps, kwargs['state_rep'],
                                 kwargs['state_transform'], kwargs['inv_state_transform'],
                                 FLAGS.standardize)

    algo = kwargs.pop('algo')
    assert algo == 'sac', "other RL algos not yet supported"

    kwargs['gnl'] = FLAGS.gnl

    watermark = "{}_soft_eq{}_wd{}_ceq{}_cwd{}_g{}".format(
        FLAGS.env_name+FLAGS.add_envname,
        FLAGS.equiv, FLAGS.wd,
        FLAGS.cequiv, FLAGS.cwd,
        FLAGS.group
    )
    wandb.init(
        project="RL, rpp vs softemlp",
        name=watermark,
        mode="disabled" if FLAGS.logoff else "online"
    )
    wandb.config.update(FLAGS)
    wandb.config.update(kwargs)

    if algo == 'sac':
        agent = SoftSACLearner(FLAGS.seed,
                               env.observation_space.sample()[np.newaxis],
                               np.asarray(env.action_space.sample())[None],
                               actor_basic_wd=float(FLAGS.wd),
                               actor_equiv_wd=[float(e) for e in FLAGS.equiv.split(',')],
                               critic_basic_wd=float(FLAGS.cwd),
                               critic_equiv_wd=[float(e) for e in FLAGS.cequiv.split(',')],
                               standardizer=replay_buffer.running_stats.standardize if FLAGS.standardize else None,
                               clipping=FLAGS.clipping,
                               gan_betas=FLAGS.gan_betas,
                               tau=FLAGS.tau, **kwargs)
        policy_mean_fn = jit(lambda p, x: agent.sac.actor.apply_fn.apply(
            {'params': p}, x)._distribution._loc)
    else:
        raise NotImplementedError()

    @jit
    def reprhos(x):
        gs = kwargs['symmetry_group'][0].samples(x.shape[0])
        ring = vmap(kwargs['state_rep'].rho_dense)(gs)
        routg = vmap(kwargs['action_rep'].rho_dense)(gs)
        return ring, routg

    equivlength = len(kwargs['symmetry_group'])
    statelength = equivlength + 1
    top_valid_returns = [-float('inf') for _ in range(statelength)]

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask,
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

        if i >= FLAGS.start_training:
            for _ in range(FLAGS.ncritic-1):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info = agent.update(batch, update_policy=False)
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            if (i//FLAGS.eval_interval) % 8 == 0:  # only do equivariance calc ever 4 evals
                eval_stats = rpp_evaluate(
                    agent, policy_mean_fn, eval_env, FLAGS.eval_episodes, kwargs, reprhos)
            else:
                eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

            wandb.log(step=info['total']['timesteps'],data={
                "avg_return": eval_stats['return'],
                "state": agent.get_current_state()
            })

        if i% FLAGS.search_interval == 0:
            current_state = agent.get_current_state()
            stats_list = softemlp_evaluate(
                agent, eval_env, FLAGS.eval_episodes, statelength) 
            return_list = [stat['return'] for stat in stats_list]
            for j in range(statelength):
                if top_valid_returns[j] < return_list[j]:
                    top_valid_returns[j] = return_list[j]
            optimal_state = max(range(statelength),
                                key=lambda idx: top_valid_returns[idx])-1
            agent.set_state(optimal_state)
            if current_state != optimal_state:
                print(f"[#{i}] {'vs'.join([f'{v:.1f}' for v in return_list])}")
                print(f"[#{i}] STATE {current_state}->{optimal_state}")


if __name__ == '__main__':
    app.run(main)
