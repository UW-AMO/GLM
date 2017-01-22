workspace()
using MLBase

# functions
compute_center(X::Matrix{Float64}) = vec(mean(X, 2))

compute_rmse(c::Vector{Float64}, X::Matrix{Float64}) =
    sqrt(mean(sum(abs2(X .- c),1)))

# data
const n = 200
const data = [2., 3.] .+ randn(2, n)

# cross validation
scores = cross_validate(function (inds) inds = inds;return(compute_center(data[:, inds])); end,        # training function
    (c, inds) -> compute_rmse(c, data[:, inds]),  # evaluation function
    n,              # total number of samples
    Kfold(n, 5))    # cross validation plan: 5-fold

# get the mean and std of the scores
(m, s) = mean_and_std(scores)
