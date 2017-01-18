workspace()
using DataFrames
using Optim
include("util_exp.jl")

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

x_init = zeros(order + 1)
g = zeros(order+1)
f = f_exp_val(x_init, params)
f_exp_grad!(g, x_init, params)


# solve Xin's problem!!
# myF = TwiceDifferentiableFunction((x)->f_exp_val(x,params),
#                                   (x,g)->f_exp_grad!(g,x,params),
#                                   (x,h)->f_exp_hess!(h,x,params))
myF = DifferentiableFunction((x)->f_exp_val(x,params),
                                  (x,g)->f_exp_grad!(g,x,params))

#myF = DifferentiableFunction((x)->f_Poisson_val(x,params), (x,g)->f_Poisson_grad!(g,x,params))

profile = false

if profile
   Profile.clear()
   @profile results = Optim.optimize(myF, x_init, BFGS())
   Profile.print(format = :flat)

else

  @time  for i = 1:1000
         results = optimize(myF, rand(size(x_init)), BFGS(), Optim.Options(x_tol = 1e-5, f_tol =1e-5))
#                                  Optim.Options(rel_tol = 1e-10, abs_tol = 1e-10))
        end
end
results = Optim.optimize(myF, randn(size(x_init)), BFGS(),Optim.Options(x_tol = 1e-5, f_tol =1e-5))
intercept = exp(results.minimizer[1])
