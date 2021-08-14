using DrWatson
@quickactivate "JuliaRL"
using ReinforcementLearning
exp = E`JuliaRL_BasicDQN_CartPole`
exp.description
env = CartPoleEnv()
state_space(env)

#-
policy = RandomPolicy()
RandomPolicy([1,3])
policy(state(env))
hook = TotalRewardPerEpisode()
run(exp.policy, env, StopAfterEpisode(10), hook)

ep = Episode(env, policy)

using Plots
plot(hook.rewards)
