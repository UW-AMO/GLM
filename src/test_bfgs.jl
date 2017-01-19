workspace()
using Optim
using Convex
using SCS
include("util_exp.jl")
include("kfold_cv.jl")

# we are testing the standard error of the sample mean for AR1 with N(0,1) innovation
phi = 0.0
mu = 0
sd = 1

ndata = 10

data = simulate_AR1(ndata, phi)
spec, freq = myperiodogram(data)

m = length(freq) # number of rows
order = 7
λ = 1
α = 0.5

myMat = ones(m,order+1)
for  ii = 1:order
  myMat[:,ii+1] = freq.^ii
end
# comment: myMat is X, spec is y.

params = exp_params()
params.myMat = myMat
params.spec = spec
params.λ = λ
params.α = α

# use bfgs to solve the glm problem with exponential distribution
res_bfgs = fit_glm_lasso_exp(params)

# use Convex.jl to solve the same problem, the solver is SCS

# Create a (column vector) variable of size n x 1.
x = Variable(order + 1)

# The problem is to minimize ||Ax - b||^2 subject to x >= 0
# This can be done by: minimize(objective, constraints)
problem = minimize(sum((params.spec).*exp(params.myMat*x) - params.myMat*x)+ params.λ*(params.α*norm(x,1) +0.5*(1-params.α)*norm(x)^2))

# Solve the problem by calling solve!
solve!(problem)

# Check the status of the problem
problem.status # :Optimal, :Infeasible, :Unbounded etc.

# Get the optimum value
problem.optval
