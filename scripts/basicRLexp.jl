using DrWatson
@quickactivate "JuliaRL"
using Reinforce, Plots
using Reinforce.MountainCarEnv: MountainCar

# define the policy and environment
struct RandomPolicy <: AbstractPolicy end
Reinforce.action(π::RandomPolicy, r, s, A) = rand(A)
π = RandomPolicy()

# Deterministic policy that is solving the problem
struct BasicCarPolicy <: Reinforce.AbstractPolicy end

Reinforce.action(policy::BasicCarPolicy, r, s, A) = s.velocity < 0 ? 1 : 3

# ep = Episode(env, π)
# for (s, a, r, s′) in ep
#     # do some custom processing of the sars-tuple
# end
# R = ep.total_reward
# T = ep.niter

env = MountainCar()

function episode!(env, π=RandomPolicy())
    ep = Episode(env, π)
    for (s, a, r, s′) in ep
        gui(plot(env))
    end
    ep.total_reward, ep.niter
end

# Main part
R, n = episode!(env, RandomPolicy())
println("reward: $R, iter: $n")

# This one can be really long...
R, n = episode!(env, BasicCarPolicy())
println("reward: $R, iter: $n")