# function to fit glm-EN model using cross validation
# params is a struct
# params.order
function glm_en_kfold_cv(params::exp_params;k::Int64 = 10, ncore = 1)
    ndata = length(params.spec)
    ntest = Int64(floor(ndata/k))
    myparams = exp_params()
    myparams.α = params.α
    myparams.λ = params.λ
    RMSEs = zeros(k)
    coeffs = zeros(k, size(params.myMat, 2))
    for i in 1:k
        # Compute the indices of the training and test set
        train_idx = [collect(1:(i-1)*ntest);collect(i*ntest+1:ndata)]
        test_idx = collect(((i-1)*ntest+1):(i*ntest))

        # build myparams
        myparams.myMat = params.myMat[train_idx, :]
        myparams.spec = params.spec[train_idx]

        # fit glm-en model using the training set
        coeffs[i, :] = fit_prox_glm_lasso_exp(myparams)

        # compute the predicted spec
        spec_predict = predict_glm_lasso_exp(params.myMat[test_idx, :], coeffs[i, :])

        # compute the RMSE on the test set
        RMSEs[i] = sum((params.spec[test_idx] - spec_predict).^2) .^ 0.5
    end
    return(mean(RMSEs))
end

# function to build a grid of λs and find the best one
function glm_en_search_lambda(X, y; alpha = 1, epsilon = 0.001, nlambda = 100, k = 10, ncore = 1)
    ndata = length(y)
    nvar = size(X, 2)
    lambda_max = findmax(X'*y)[1]/ndata/alpha
    lambda_min = epsilon * lambda_max
    lambdas = exp(linspace(log(lambda_min), log(lambda_max), nlambda))
    errors = zeros(nlambda)
    myparams = exp_params()
    myparams.myMat = X
    myparams.spec = y
    myparams.α = alpha
    for i in 1:nlambda
        # for each lambda, run cross validation to estimate the error in test set
        myparams.λ = lambdas[i]
        errors[i] = glm_en_kfold_cv(myparams, k = k, ncore = ncore)
    end
    lambda_best = lambdas[findmin(errors)[2]]
    myparams.λ = lambda_best
    return([fit_prox_glm_lasso_exp(myparams), lambda_best])
end
