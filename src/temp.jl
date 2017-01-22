workspace()
using Optim
using Convex
using SCS
using Distributions
using MLBase
include("util_exp.jl")
include("kfold_cv.jl")

# # we are testing the standard error of the sample mean for AR1 with N(0,1) innovation
# phi = 0.3
# mu = 0
# sd = 1

ndata = 10
order = 4
nsim = 500
λ = 0
α = 0.5
xtrue = [1, 1, 1, 1]/10

# data = simulate_AR1(ndata, phi)
# spec, freq = myperiodogram(data)


# myMat = Atrue

# m = length(spec) # number of rows


# myMat = ones(m,order+1)
# for  ii = 1:order
#   myMat[:,ii+1] = freq.^ii
# end
# comment: myMat is X, spec is y.

params = exp_params()
params.λ = λ
params.α = α
res = zeros(nsim, order)

# do MC simulation to see if the correct
for i in 1:nsim
    params.myMat = rand(ndata, order)
    params.spec  = simulate_glm_exp(xtrue, params.myMat)
    res[i, :] = fit_prox_glm_lasso_exp(params)
    println(res[i, :])
end
println("The true coefficients are $(round(xtrue, 3))")
println("The average of the $(nsim) estimated coefficients are $(round(mean(res, 1), 3))")

# res_cv = glm_en_kfold_cv2(params, 10)
# print(res_cv)


# # prox_fit = fit_prox_glm_lasso_exp(params)
# res = glm_en_search_lambda(params.myMat, params.spec)
# predicted_spec = predict_glm_lasso_exp(params.myMat, res[1])
# println("The RMSE is $(sqrt(mean(abs2(params.spec .- predicted_spec))))")
# mean_and_std(params.spec)
