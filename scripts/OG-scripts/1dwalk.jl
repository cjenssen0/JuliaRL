using DrWatson
@quickactivate "JuliaRL"
using ReinforcementLearning, Plots

# Define env and policy
env = CartPoleEnv()
S = state_space(env)
A = action_space(env)
is_terminated(env)
reward(env)

# Simple run-through with random policy
Π = RandomPolicy()
Π = AdHocPolicy()
stopCond = StopAfterEpisode(100)
hook = TotalRewardPerEpisode()
#run(
           #Π,
          # env,
         #  stopCond,
        #   hook
       #)

# Manual method
function run_algo()
    env = CartPoleEnv()
    #Π = RandomPolicy()
    Π = AdHocPolicy()
    stopCond = StopAfterEpisode(10)
    hook = TotalRewardPerEpisode()
    anim = Animation()
    for i in 1:1
        reset!(env)
        Π(env)
        p = plot()
        while !is_terminated(env)
            action = Π(env)
            env(action)
            #push!(p, plot(env))
            plot(env)
            #display(Scene(env))
            frame(anim)
            stopCond(Π, env) && return
        end
    end
    return anim
end
anim = run_algo()
gif(anim)