workspace()
using Distributions
using Optim, LineSearches
using Convex
using SCS
include("util_exp.jl")
include("kfold_cv.jl")

# generate exponentially distributed samples with identity link function

nsim = 1
order = 4
ndata = 10
# freq = collect(linspace(0.01,0.5,ndata))
# Atrue = rand(ndata, nvar)
# myMat = ones(ndata, order+1)
# for  ii = 1:order
#   myMat[:,ii+1] = freq.^ii
# end
myMat = randn(ndata, order + 1)
xtrue = randn(order + 1)
thetas = exp(myMat*xtrue)


λ = 0
α = 0.5


##### Test BFGS method with log link
params = exp_params()
params.myMat = myMat
params.λ = λ
params.α = α


x_bfgs = zeros(order+1)
x_prox = zeros(order+1)
x_SCS = zeros(order + 1)
for i in 1:nsim
    println(i)
    params.spec = map(x -> rand(Exponential(x)), thetas)

    # BFGS
    params.fval = f_exp_val
    params.gval = f_exp_grad!
    x_bfgs += fit_glm_lasso_exp(params)

    # prox gradient
    params.fval = f_exp_val_smooth
    params.gval = f_exp_grad_smooth!
    x_prox += fit_prox_glm_lasso_exp(params)


    # SCS
    x = Variable(order + 1)
    problem = minimize(sum((params.spec).*exp(params.myMat*x) - params.myMat*x) + params.λ*(params.α*norm(x,1) +0.5*(1-params.α)*sumsquares(x)))
    solve!(problem, SCSSolver(verbose=false), verbose = false)
    x_SCS += x.value
end



println("The true coefficients are $xtrue")
println("The estimated coefficients from BFGS are $((x_bfgs)./nsim)")
println("The estimated coefficients from Prox Gradient are $((x_prox)./nsim)")
println("The estimated coefficients from SCS are $((x_SCS)./nsim)")
