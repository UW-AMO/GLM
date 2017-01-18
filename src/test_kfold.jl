workspace()
using DataFrames
using Optim
include("util_exp.jl")
include("kfold_cv.jl")

df = readtable("../data/AR1.csv")
freq = df[:,2]
spec = df[:,3]
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

res = fit_glm_lasso_exp(params)
res1 = predict_glm_lasso_exp(params.myMat, res)
res2 = glm_en_kfold_cv(params)

res3 = glm_en_search_lambda(params.myMat, params.spec, params.α)
