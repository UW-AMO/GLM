workspace()
using DataFrames
using Optim, LineSearches
using Convex
using SCS
include("util_exp.jl")
include("kfold_cv.jl")

myMat = convert(Matrix{Float64}, readtable("../data/VandermondeMatrix.csv"))
specs = convert(Matrix{Float64}, readtable("../data/b_VandermondeMatrix.csv"))
spec = specs[:,1]
X_true = convert(Matrix{Float64}, readtable("../data/trueSol.csv"))
x_true = X_true[:,1]
X0 = convert(Matrix{Float64}, readtable("../data/startSol.csv"))
x0 = X0[:,1]
m = length(spec) # number of rows
order = size(myMat, 2)-1
λ = 0.2
α = 1.0

params = exp_params()
params.myMat = myMat
params.spec = spec
params.λ = λ
params.α = α

# primal BFGS
params.fval = f_exp_val
params.gval = f_exp_grad!
x_bfgs = fit_glm_lasso_exp(params)

dualBFGS = false
if dualBFGS
    # dual bFGS
    params.fval = f_exp_val_dual
    params.gval = f_exp_dual_grad!
    x_from_dual = fit_glm_lasso_exp_dual(params)
end

# via prox
params.fval = f_exp_val_smooth
params.gval = f_exp_grad_smooth!
x_prox = fit_prox_glm_lasso_exp(params)
# use Convex.jl to solve the same problem, the solver is SCS

# primal BFGS id link
params.fval = f_exp_val_id
params.gval = f_exp_grad_id!
x_new = fit_glm_lasso_exp(params)


doConvex = true

if doConvex
    # Create a (column vector) variable of size n x 1.
    x = Variable(order + 1)

    # The problem is to minimize ||Ax - b||^2 subject to x >= 0
    # This can be done by: minimize(objective, constraints)
    problem = minimize(sum((params.spec).*exp(params.myMat*x) - params.myMat*x) + params.λ*(params.α*norm(x,1) +0.5*(1-params.α)*sumsquares(x)))

    # Solve the problem by calling solve!
    solve!(problem)
    println(x.value)
    # Check the status of the problem
    problem.status # :Optimal, :Infeasible, :Unbounded etc.
    # Get the optimum value
    problem.optval
    @printf("Relative error of bfgs solution: %7.3e\n", norm(x.value - x_bfgs)/norm(x.value))
    @printf("Relative error first bfgs coefficient: %7.3e\n", abs(x.value[1] - x_bfgs[1])/abs(x.value[1]))

    if dualBFGS
        @printf("Relative error of dual solution: %7.3e\n", norm(x.value - x_from_dual)/norm(x.value))
        @printf("Relative error dual first coefficient: %7.3e\n", abs(x.value[1] - x_from_dual[1])/abs(x.value[1]))
        #@printf("Relative error dual first coefficient: %7.3e\n", abs(x.value[1] - x_form_dual[1])/abs(x.value[1]))
    end


    @printf("Relative error of prox solution: %7.3e\n", norm(x.value - x_prox)/norm(x.value))
    @printf("Relative error prox first coefficient: %7.3e\n", abs(x.value[1] - x_prox[1])/abs(x.value[1]))

    println("Convex.JL")
    println(round(x.value,2))
end
println("Prox")
println(round(x_prox,2))
println("BFGS")
println(round(x_bfgs,2))
if dualBFGS
    println("BFGS Dual")
    println(round(x_from_dual,2))
end
println("True Params")
println(round(x_true,2))
#println("BFGS (ID link)")
#println(round(x_new[1],2))
#println(x_bfgs)
#println(x.value)
