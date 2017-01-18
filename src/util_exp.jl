#import ArrayViews: view, StridedView, ContiguousView

type exp_params
    myMat::Matrix{Float64}
    spec::Vector{Float64}
    λ::Float64
    α::Float64
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
  # rs = params.myMat*x
  # ers = exp(rs).*params.spec
  # m = length(params.spec)
   #vA =  ContiguousView{Float64,1,Array{Float64,2}}[view(params.myMat',:,ii) for ii=1:m]

   #gradient of the smooth part
   #myg = zeros(size(g))

   # don't need any of that crap, this is faster:
#   myg = params.myMat'*(ers - 1)
  #  for ii = 1:m
  #    myg += params.spec[ii]*ers[ii]*params.myMat[ii,:]
  #  end
#   println(size(sum(params.myMat',2)))
#   println(g)
#   myg -= sum(params.myMat',2)

   # gradient of the elastic net
  # myg += params.λ*(params.α*sign(x) + 0.5*(1-params.α)*x)
  # copy!(g,myg)
  copy!(g, params.myMat'*(exp(params.myMat*x).*params.spec -1.0) + params.λ*(params.α*sign(x) + 0.5*(1-params.α)*x))
end

function f_exp_hess!(h::Matrix{Float64}, x::Vector{Float64}, params)
   rs = params.myMat*x
   ers = (params.spec).*exp(rs)
   copy!(h, params.myMat'*diagm(ers)*params.myMat+params.λ*(1-params.α)*eye(length(x)))
end

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

# fit glm lasso model with exponential distribution

function fit_glm_lasso_exp(params::exp_params)
    nvar = size(params.myMat, 2)
    x_init = zeros(nvar)
    g = zeros(nvar)
    f = f_exp_val(x_init, params)
    f_exp_grad!(g, x_init, params)
    myF = DifferentiableFunction((x)->f_exp_val(x,params),
                                      (x,g)->f_exp_grad!(g,x,params))
    results = Optim.optimize(myF, x_init, BFGS())
    return(results.minimizer)
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
    return([tmp, freqs])
end

function se_glm_lasso(data; order = 5, alpha = 1, nlambda = 100, k = 10, epsilon = 0.001, ncore = 1)
    N = length(data)

    # step 1: compute the periodograms and corresponding frequencies
    spec, freq = myperiodogram(data)

    # remove the data point corresponding to frequency zero
    spec = spec[2:end]
    freq = freq[2:end]

    # step 2: use GLM with EN regularization to fit polynomial

    myMat = ones(length(freq), order+1)
    for  ii = 1:order
      myMat[:,ii+1] = freq.^ii
    end

    res, lambda_best = glm_en_search_lambda(myMat, spec, alpha = alpha, epsilon = epsilon, nlambda = nlambda, k = k, ncore = ncore)
    p0_hat = exp(res[1])
    println(p0_hat)

    # step 3: return the estimated standard error
    return(sqrt(p0_hat/N))
end
