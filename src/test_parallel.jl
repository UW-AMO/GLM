workspace()
@everywhere using Optim
@everywhere using DataFrames
@everywhere include("util_exp.jl")
@everywhere include("kfold_cv.jl")


# we are testing the standard error of the sample mean for AR1 with N(0,1) innovation
phi = 0.2
mu = 0
sd = 1


# method 1: using Monte Carlo
nsim = 100
ndata = 100


means = zeros(nsim)
for i in 1:nsim
    data = simulate_AR1(ndata, phi)
    means[i] = mean(data)
end
sd_mc = std(means)



# method 2: using our method


@time se_glm = @parallel (vcat) for i in 1:nsim
    data_raw = simulate_AR1(ndata, phi)
    data = mu_IF(data_raw)
    se_glm_lasso(data)
end

# println("The theoretical StdDev is $(sd/sqrt(ndata))")
println("The standard error using Monte Carlo is $sd_mc")
println("The standard error estimated by se_glm_lasso is $(mean(se_glm))")



# compare results
