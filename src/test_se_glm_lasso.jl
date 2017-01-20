workspace()
using Optim
using DataFrames
include("util_exp.jl")
include("kfold_cv.jl")



# we are testing the standard error of the sample mean for AR1 with N(0,1) innovation
phi = 0.2

# we are testing the standard error of the sample mean for iid N(0,1)
phi = 0.3

mu = 0
sd = 1


# method 1: using Monte Carlo
nsim = 10
ndata = 100


means = zeros(nsim)
for i in 1:nsim
    data = simulate_AR1(ndata, phi)
    means[i] = mean(data)
end
sd_mc = std(means)



# method 2: using our method

profile = true

se_glm = zeros(nsim)

@time for i in 1:nsim

  println(i)
  if profile
    Profile.clear()

      @profile for i in 1:nsim

        data_raw = simulate_AR1(ndata, phi)
        data = mu_IF(data_raw)
        se_glm[i] = se_glm_lasso(data)
      end
  end
end


# println("The theoretical StdDev is $(sd/sqrt(ndata))")

Profile.print(format = :flat)

println("The theoretical StdDev is $(sd/sqrt(ndata))")

println("The standard error using Monte Carlo is $sd_mc")
println("The standard error estimated by se_glm_lasso is $(mean(se_glm))")



# compare results
