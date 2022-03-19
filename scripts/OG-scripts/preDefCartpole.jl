using ReinforcementLearning, Plots, Flux

# struct RandomPolicy <: AbstractPolicy end
# action(π::RandomPolicy, r, s, A) = rand(A)
struct AdHocPolicy <: AbstractPolicy end
function action(π::AdHocPolicy, s)
    if s[4] < 0
        action = 1
    else
        action = 2
    end
    return action
end

(π::AdHocPolicy)(env) = action(π, state(env))
state(env)

# Initialize setup
# experiment = E`JuliaRL_BasicDQN_CartPole`
env = CartPoleEnv()
# p = RandomPolicy(env)
p = AdHocPolicy()
a = p(env)

# defining agent, stopping conditions and metrics
agent = Agent(
    # policy=RandomPolicy(env)
    policy = AdHocPolicy(),
    trajectory = VectorSARTTrajectory()
)
stopCond = StopAfterEpisode(100)
hook = TotalRewardPerEpisode()

# Running the experiments
# run(experiment.agent, env, stopCond, hook)
run(agent, env, stopCond, hook)

## Plotting return
begin
    plot(hook.rewards, lw=2)
    xlabel!("ep"); ylabel!("R")
end

run(
    agent,
    env,
    stopCond,
    # DoEveryNStep(50) do t, agent, env
    DoEveryNEpisode(50) do t, agent, env
    # Flux.testmode!(agent)
    run(agent, env, stopCond, hook)
    # Flux.trainmode!(agent)
end
    )