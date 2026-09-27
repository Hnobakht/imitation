"""Sample script demonstrating AIRL algorithm with PySIDNy (SINDyRewardNet) reward network."""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import SINDyRewardNet
from imitation.util.util import make_vec_env

def main():
    rng = np.random.default_rng(0)
    env_name = "seals:seals/CartPole-v0"

    print(f"Creating environment {env_name}...")
    venv = make_vec_env(
        env_name,
        rng=rng,
        n_envs=2,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )

    print("Generating expert demonstrations using a synthetic expert policy...")
    expert_policy = PPO(
        policy=MlpPolicy,
        env=venv,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.003,
        n_epochs=5,
        n_steps=64,
    )
    expert_policy.learn(total_timesteps=1_000)

    print("Sampling expert transitions...")
    rollouts = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=10),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)

    print("Creating SINDyRewardNet for AIRL...")
    sindy_reward_net = SINDyRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_state=True,
        use_action=True,
        use_next_state=False,
        use_done=False,
        degree=2,
        threshold=1e-4,
    )

    generator = PPO(
        policy=MlpPolicy,
        env=venv,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.001,
        n_epochs=5,
        n_steps=64,
    )

    print("Initializing AIRL trainer with SINDy reward network...")
    airl_trainer = AIRL(
        demonstrations=transitions,
        demo_batch_size=64,
        venv=venv,
        gen_algo=generator,
        reward_net=sindy_reward_net,
    )

    print("Evaluating initial generator policy...")
    mean_reward, _ = evaluate_policy(airl_trainer.policy, venv, n_eval_episodes=5)
    print(f"Initial policy mean reward: {mean_reward:.2f}")

    print("Training AIRL with SINDyRewardNet...")
    airl_trainer.train(total_timesteps=512)

    print("Fitting PySINDy model on transitions to extract explicit sparse equations...")
    batch_size = min(len(transitions.obs), 128)
    sample_obs = transitions.obs[:batch_size]
    sample_acts = transitions.acts[:batch_size]
    sample_next_obs = transitions.next_obs[:batch_size]
    sample_dones = transitions.dones[:batch_size]

    logits = sindy_reward_net.predict(
        sample_obs, sample_acts, sample_next_obs, sample_dones
    )
    sindy_reward_net.fit_sindy(
        sample_obs, sample_acts, sample_next_obs, sample_dones, targets=logits
    )

    feature_names = sindy_reward_net.feature_library.get_feature_names()
    coefs = sindy_reward_net.sindy_model.coefficients()
    print("Identified PySINDy Reward Network Features and Coefficients:")
    for fn, coef in zip(feature_names, coefs.flatten()):
        if abs(coef) > 1e-4:
            print(f"  {fn}: {coef:.4f}")

    print("Evaluating policy after AIRL training...")
    mean_reward, _ = evaluate_policy(airl_trainer.policy, venv, n_eval_episodes=5)
    print(f"Post-training mean reward: {mean_reward:.2f}")

if __name__ == "__main__":
    main()
