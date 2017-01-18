workspace()
using Optim
using DataFrames
include("kfold_cv.jl")
include("util_exp.jl")

# we are testing the standard error of the sample mean for iid N(0,1)
phi = 0
mu = 0
sd = 1

# theoretically, the standard error should be 1


# method 1: using Monte Carlo



# method 2: using our method



# compare results
