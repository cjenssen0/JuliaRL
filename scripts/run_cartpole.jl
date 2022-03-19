using DrWatson
@quickactivate "JuliaRL"

using ReinforcementLearning

# Load policies for discrete environments
includet(srcdir("DiscretePolicies.jl"))
includet(srcdir("Transformations.jl"))

## Define environment and its attributes
env = CartPoleEnv()
S = state_space(env)
A = action_space(env)
nS, nA = map(length, (S,A))

# Define discretized state-space version of cartpole
nbins = 5
env_discrete = StateTransformedEnv(
               env;
               state_mapping = s -> state_mapping(s, nbins=nbins),
               state_space_mapping = _ -> Base.OneTo(nbins)
           )

# init stuff - get state from env, action from policy
s = state(env)
a = rand(A)

#Act on env, get reward, done-signal and next state s_next
env(a)
s_next = state(env)
r = reward(env)
done = is_terminated(env)

# Define the policy and experiment parameters
Π = AdHocPolicy()
n_episodes = 100
capacity = 10
stop_condition = StopAfterEpisode(n_episodes)
hook = TotalRewardPerEpisode()

## Based on `run` in ReinforcementLearning.jl
function _run(policy::AbstractPolicy, env::AbstractEnv, env_discrete::AbstractEnv, stop_condition, hook::AbstractHook)
    
    # run until stop condition is met
    is_stop = false
    while !is_stop

        # set initial state
        reset!(env)

        # Do pre episode stuff with policy (and hook)
        policy(PRE_EPISODE_STAGE, env)
        hook(PRE_EPISODE_STAGE, policy, env)

        # run the episode until reaching a terminal state
        while !is_terminated(env)

            # Do something with policy before acting
            policy(PRE_ACT_STAGE, env)

            # Get action from policy
            a = policy(env_discrete)
            s = copy(state(env))
            s_discrete = state(env_discrete)
            hook(PRE_ACT_STAGE, policy, env)

            # Act on the environment
            env(a)
            s_next = state(env)
            r = reward(env)
            d = is_terminated(env)

            # Do stuff after applying action on the environment
            policy(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, policy, env)

            if stop_condition(policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            # TODO do something special every nth step:
            policy(POST_EPISODE_STAGE, env)  # let the policy see the last observation
            hook(POST_EPISODE_STAGE, policy, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, policy, env)
end

# defining agent, stopping conditions and metrics
agent = Agent(
    policy = AdHocPolicy(),
    trajectory = CircularArraySARTTrajectory(capacity=capacity, state = Vector{Float64} => (nS,))
    # trajectory = VectorSARTTrajectory()
)
# Run both as "agent" and policy
_run(agent, env, env_discrete, stop_condition, hook)
_run(Π, env, env_discrete, stop_condition, hook)
print("Total reward over $n_episodes episodes: $(sum(hook.rewards))")