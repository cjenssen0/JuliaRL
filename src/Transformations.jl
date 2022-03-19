using DrWatson
@quickactivate "JuliaRL"

"""
Find the index of the closest value for x in A
"""
findnearest(A, x) = argmin(abs.(A .- x))

"""
Mapping from continuous state-space to discrete for cartpole environment.
For now, hardcoded bins...
"""
function state_mapping(s)
    s_bin1 = range(-3, 3, length=nbins)
    s_bin2 = range(-100, 100, length=nbins)
    s_bin3 = range(-0.3, 0.3, length=nbins)
    s_bin4 = range(-100, 100, length=nbins)
    s_bins = [s_bin1, s_bin2, s_bin3, s_bin4]
    obs = [findnearest(s_bin, s[i]) for (i,s_bin) in enumerate(s_bins)]
    return obs
end

"""
Mapping from continuous state-space to discrete for cartpole environment. s_bins is a vector of s_bins; binning values for each state-component. Assumes equal length for each state-component.
"""
function state_mapping(s, s_bins)
    obs = [findnearest(s_bin, s[i]) for (i,s_bin) in enumerate(s_bins)]
    return obs
end