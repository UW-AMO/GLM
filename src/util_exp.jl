#import ArrayViews: view, StridedView, ContiguousView

type exp_params
    myMat::Matrix{Float64}
    spec::Vector{Float64}
    fval::Function
    gval!::Function
    λ::Float64
    α::Float64
    iter_max::Int64
    tol::Float64
    exp_params() = new()
end
# \sum_{i} y_i \exp(<a_i, x>) - 1^TAx + λ(α||x||_1 + (1-α)/2 ||x||²)
function f_exp_val(x::Vector{Float64}, params)
    #BLAS.axpy!(params.myMat,x,rs)
    #rs = BLAS.gemv('N', 1.0, params.myMat, x)
    rs = params.myMat*x
    fs = (params.spec).*exp(rs) - rs
    f = sum(fs)+ params.λ*(params.α*norm(x,1) +0.5*(1-params.α)*norm(x)^2)
   return f
end
function f_exp_grad!(g::Vector{Float64}, x::Vector{Float64}, params)
  copy!(g, params.myMat'*(exp(params.myMat*x).*params.spec -1.0) + params.λ*(params.α*sign(x) + (1-params.α)*x))
end
function soft_thresh(x::Vector{Float64},λ::Float64)
   return x-min(max(x,-λ),λ)
end
function inf_proj(x::Vector{Float64},λ::Float64)
   return min(max(x,-λ),λ)
end

function prox_enet(x::Vector{Float64}, γ::Float64, params)
  a1 = params.λ*params.α
  a2 = params.λ*(1.0-params.α)
  return soft_thresh(x/(1+a2*γ), a1*γ/(1+a2*γ))
end

#################################################################################
# GLM functions
#################################################################################
function f_exp_val_smooth(x::Vector{Float64}, params)
    #BLAS.axpy!(params.myMat,x,rs)
    #rs = BLAS.gemv('N', 1.0, params.myMat, x)
    rs = params.myMat*x
    fs = (params.spec).*exp(rs) - rs
    f = sum(fs)
   return f
end
function f_exp_grad_smooth!(g::Vector{Float64}, x::Vector{Float64}, params)
  copy!(g, params.myMat'*(exp(params.myMat*x).*params.spec -1.0))
  return mean(exp(params.myMat*x))
end
function f_exp_val_smooth_id(x::Vector{Float64}, params)
    #BLAS.axpy!(params.myMat,x,rs)
    #rs = BLAS.gemv('N', 1.0, params.myMat, x)
    rs = params.myMat*x
    if minimum(rs) > 0
      return -sum(log(rs)) + dot(params.spec, rs)
    else
      return Inf
    end
end
function f_exp_grad_smooth_id!(g::Vector{Float64}, x::Vector{Float64}, params)
  copy!(g, params.myMat'*(-1.0./(params.myMat*x) + params.spec))
  return min(1./minimum((params.myMat*x).^2),1.0e4)
end
function f_exp_val_id(x::Vector{Float64}, params)
    #BLAS.axpy!(params.myMat,x,rs)
    #rs = BLAS.gemv('N', 1.0, params.myMat, x)
    rs = params.myMat*x
    if minimum(rs) > 0
      return -sum(log(rs)) + dot(params.spec, rs) + params.λ*(params.α*norm(x,1) +0.5*(1-params.α)*norm(x)^2)
    else
      return Inf
    end
end
function f_exp_grad_id!(g::Vector{Float64}, x::Vector{Float64}, params)
  copy!(g, params.myMat'*(-1.0./(params.myMat*x) + params.spec)+ params.λ*(params.α*sign(x) + (1-params.α)*x))
  return min(1./minimum((params.myMat*x).^2),1.0e4)
end

function f_exp_val_dual(z,params)
  a₁ = params.λ*params.α
  a₂ = params.λ*(1.0-params.α)
  if minimum(z) > 0
    return (1/(2*a₂))*norm(soft_thresh(params.myMat'*(1.0-z),a₁))^2 + sum(z.*(log(z./params.spec) -1.0))
  else
    return Inf
  end

end
function f_exp_dual_grad!(g::Vector{Float64}, z::Vector{Float64}, params)
  a₁ = params.λ*params.α
  a₂ = params.λ*(1.0-params.α)
  copy!(g, -(1/a₂)*params.myMat*soft_thresh(params.myMat'*(1.0-z), a₁)+ log(z./params.spec))
end
function primal_from_dual(z::Vector{Float64}, params)
  a₁ = params.λ*params.α
  a₂ = params.λ*(1.0-params.α)
    return (1/a₂)*soft_thresh(params.myMat'*(1.0-z),a₁)
end
#############################################################################################
# Simulation functions
#############################################################################################
function simulate_AR1(n, phi; x0 = 0, mu = 0, sd = 1)
    ts = zeros(n+1)
    ts[1] = x0
#    epsilons = randn(n) * sd + mu
    for i in 1:n
        ts[i+1]= ts[i]*phi + randn() * sd + mu
    end
    return(ts[2:end])
end
function mu_IF(data::Vector{Float64})
    mu = mean(data)
    return(data - mu)
end

###############################################################################
# Optimization routines
###############################################################################

# fit glm lasso model with exponential distribution
function fit_prox_glm_lasso_exp(params::exp_params)
    nvar = size(params.myMat, 2)
    x = zeros(nvar)
    #x[1] = 2.0
    x_old = zeros(nvar)
    aNorm2 = vecnorm(params.myMat)^2
    g = zeros(nvar)
    f_val = x->params.fval(x,params) #x-> f_exp_val_smooth(x, params)
    g_val! = (x,g)->params.gval!(x,g,params) #(g,x) -> f_exp_grad_smooth!(g, x, params)
    prox_fun = (x,γ)->prox_enet(x,γ,params)
    Lip = g_val!(g,x)
    converged = false

    # right now γ is set arbitrarily.
    easy_γ = 0.2
    tol = params.tol
    iter_max = params.iter_max
    iter = 0
    print = true
    step_scale = 1.9
    while converged == false
      iter = iter + 1
      x_old = copy(x)
      # TODO: make sure statement below always works by using a line search.
      #assert(minimum(params.myMat*x) > 0)
      f = f_val(x)
      γ = step_scale/(Lip*aNorm2)
      #println(γ)
      x = prox_fun(x - γ*g,γ)
      Lip = g_val!(g,x)
      res = (x-x_old)/γ
      converged = (norm(res) < tol || iter > params.iter_max)
      if print
        @printf("iter: %d, val: %7.2e, conv: %7.2e\n", iter, f, norm(res))
      end
    end
    return x
end

# use BFGS on nonsmooth function
function fit_glm_lasso_exp(params::exp_params)
    nvar = size(params.myMat, 2)
    x_init = rand(nvar)
    g = zeros(nvar)
    fval = params.fval
    gval! = params.gval!
    f = fval(x_init, params)
    gval!(g, x_init, params)
    myF = DifferentiableFunction((x)->fval(x,params),
                                      (x,g)->gval!(g,x,params))
    #algo_bt = BFGS(;linesearch = LineSearches.backtracking!)
    #algo_bt = BFGS(;linesearch = LineSearches.morethuente!)
    algo_bt = BFGS(;linesearch = LineSearches.strongwolfe!)
    #results = Optim.optimize(myF, x_init, algo_bt, Optim.Options(x_tol = 1e-3, f_tol =1e-3))
    results = Optim.optimize(myF, x_init, algo_bt)
    return(results.minimizer)
end

# use BFGS on smooth dual function, and recover primal from dual.
function fit_glm_lasso_exp_dual(params::exp_params)
    nvar = size(params.myMat, 1)
    z_init = params.spec
    g = zeros(nvar)
    fval = params.fval
    gval! = params.gval!
    f = fval(z_init, params)
    gval!(g, z_init, params)
    #myF = DifferentiableFunction((z)->f_exp_val_dual(z,params),
    #                                  (z,g)->f_exp_dual_grad!(g,z,params))
    #results = Optim.optimize(myF, x_init, BFGS(), Optim.Options(x_tol = 1e-5, f_tol =1e-3))
    myF = DifferentiableFunction((z)->fval(z,params),
                                      (z,g)->gval!(g,z,params))
   algo_bt = BFGS(;linesearch = LineSearches.backtracking!)
   #algo_bt = BFGS(;linesearch = LineSearches.strongwolfe!)
    #algo_bt = BFGS(;linesearch = LineSearches.morethuente!)

    #results = Optim.optimize(myF, x_init, BFGS(), Optim.Options(x_tol = 1e-5, f_tol =1e-3))
    results = Optim.optimize(myF, z_init, algo_bt,Optim.Options(x_tol = 1e-5, f_tol =1e-5, g_tol = 1e-5))

    println(results)
    dual = results.minimizer
  #  println(dual)
    primal = primal_from_dual(dual, params)
#    println(primal)
    return primal
end
# coeffs is the coeffcients of the GLM model
# x is the matrix of test data. each row is an example and each column is a variable
# rtype: vector of the predi
function predict_glm_lasso_exp(x::Matrix{Float64}, coeffs::Vector{Float64})
    return(exp(x*coeffs))
end


# Function to compute the periodogram of the data Vector
function myperiodogram(data)
    data_fft=fft(data)
    N=length(data)
    #   tmp=1:N
    #   inset=tmp[(1:N)<floor(N/2)]
    tmp = abs2(data_fft[1:Int64(floor(N/2))])./N
    tmp = map(x -> max(0.01,x), tmp)
    freqs = collect(0:Int64(floor(N/2)-1))./N


    # remove the data point corresponding to frequency zero
    tmp = tmp[2:end]
    freqs = freqs[2:end]
    return([tmp, freqs])
end

function se_glm_lasso(data; order = 5, alpha = 1, nlambda = 100, k = 10, epsilon = 0.001, ncore = 1)
    N = length(data)

    # step 1: compute the periodograms and corresponding frequencies
    spec, freq = myperiodogram(data)

    # step 2: use GLM with EN regularization to fit polynomial

    myMat = ones(length(freq), order+1)
    for  ii = 1:order
      myMat[:,ii+1] = freq.^ii
    end

    res, lambda_best = glm_en_search_lambda(myMat, spec, alpha = alpha, epsilon = epsilon, nlambda = nlambda, k = k, ncore = ncore)
    p0_hat = exp(res[1])
#    println(p0_hat)

    # step 3: return the estimated standard error
    return(sqrt(p0_hat/N))
end
