using DrWatson
@quickactivate "JuliaRL"

using ReinforcementLearning

"""
Discrete adhoc policy for cartpole
"""
struct AdHocPolicy <: AbstractPolicy end

"""
Define state->action mapping based on angular velocity dθ
"""
function (π::AdHocPolicy)(env)
    s = state(env)
    if s[4] < 0
        a = 1
    else
        a = 2
    end
    return a
end

"""
Define what to do in different stages of the experiment
"""
function (π::AdHocPolicy)(::PreEpisodeStage, env::AbstractEnv)
    println("Doing stuff with policy in pre episode stage")
    return nothing
end
function (π::AdHocPolicy)(::PreActStage, env::AbstractEnv)
    println("Doing stuff with policy in pre act stage")
    return nothing
end
function (π::AdHocPolicy)(::PostActStage, env::AbstractEnv)
    println("Doing stuff with policy in post act stage")
    return nothing
end
function (π::AdHocPolicy)(::PostEpisodeStage, env::AbstractEnv)
    println("Doing stuff with policy in post episode stage")
    return nothing
end