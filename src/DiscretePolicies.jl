using DrWatson
@quickactivate "JuliaRL"

using ReinforcementLearning

"""
Discrete adhoc policy for cartpole
"""
struct AdHocPolicy <: AbstractPolicy end
function action(π::AdHocPolicy, s)
    if s[4] < 0
        action = 1
    else
        action = 2
    end
    return action
end
