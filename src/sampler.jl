using LinearAlgebra
using Random
using Statistics
using ProgressMeter

struct BayesianIV
    Z::Vector{Float64}
    W::Vector{Float64}
    Y::Vector{Float64}
    X::Matrix{Float64}
    N_a::Float64
    N::Int
    dim::Int

    function BayesianIV(Z, W, Y, X, N_a)
        new(Z, W, Y, X, N_a, length(Y), size(X, 2))
    end
end

function G_sampler(model::BayesianIV, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t)
    G = zeros(model.N)
    
    # pattern for (Z_i = 0) & (W_i = 1)
    G[(model.Z .== 0) .& (model.W .== 1)] .= 0

    # pattern for (Z_i = 1) & (W_i = 0)
    G[(model.Z .== 1) .& (model.W .== 0)] .= 1

    # pattern for (Z_i = 0) & (W_i = 0)
    mask_z0_w0 = (model.Z .== 0) .& (model.W .== 0)
    X_z0_w0 = model.X[mask_z0_w0, :]
    Y_z0_w0 = model.Y[mask_z0_w0]
    N_z0_w0 = length(Y_z0_w0)
    # log p(G_i = co | Y, \theta)
    log_prob_co = -log1p.(exp.(X_z0_w0 * gamma_at) .+ exp.(X_z0_w0 * gamma_nt))
    # log p(G_i = nt | Y, \theta)
    log_prob_nt = X_z0_w0 * gamma_nt - log1p.(exp.(X_z0_w0 * gamma_at) .+ exp.(X_z0_w0 * gamma_nt))
    # log p(Y = 1, | G_i = co, \theta)
    log_prob_y1_given_co_c = X_z0_w0 * beta_co_c - log1p.(exp.(X_z0_w0 * beta_co_c))
    # log p(Y = 1, | G_i = nt, \theta)
    log_prob_y1_given_nt = X_z0_w0 * beta_nt - log1p.(exp.(X_z0_w0 * beta_nt))
    # log p(Y = 0, | G_i = co, \theta)
    log_prob_y0_given_co_c = -log1p.(exp.(X_z0_w0 * beta_co_c))
    # log p(Y = 0, | G_i = nt, \theta)
    log_prob_y0_given_nt = -log1p.(exp.(X_z0_w0 * beta_nt))
    # log p(G_i = co | Y_i, \theta)
    logp = Y_z0_w0 .* (log_prob_co + log_prob_y1_given_co_c - log.(exp.(log_prob_co + log_prob_y1_given_co_c) .+ exp.(log_prob_nt + log_prob_y1_given_nt)))
    logp += (1 .- Y_z0_w0) .* (log_prob_co + log_prob_y0_given_co_c - log.(exp.(log_prob_co + log_prob_y0_given_co_c) .+ exp.(log_prob_nt + log_prob_y0_given_nt)))
    # Deciding co or nt
    threshold = log.(rand(N_z0_w0))
    G_z0_w0 = ones(N_z0_w0)
    G_z0_w0[threshold .< logp] .= 2
    G[mask_z0_w0] = G_z0_w0

    # pattern for (Z_i = 1) & (W_i = 1)
    mask_z1_w1 = (model.Z .== 1) .& (model.W .== 1)
    X_z1_w1 = model.X[mask_z1_w1, :]
    Y_z1_w1 = model.Y[mask_z1_w1]
    N_z1_w1 = length(Y_z1_w1)
    # log p(G_i = co | Y, \theta)
    log_prob_co = -log1p.(exp.(X_z1_w1 * gamma_at) .+ exp.(X_z1_w1 * gamma_nt))
    # log p(G_i = at | Y, \theta)
    log_prob_at = X_z1_w1 * gamma_at - log1p.(exp.(X_z1_w1 * gamma_at) .+ exp.(X_z1_w1 * gamma_nt))
    # log p(Y = 1, | G_i = co, \theta)
    log_prob_y1_given_co_t = X_z1_w1 * beta_co_t - log1p.(exp.(X_z1_w1 * beta_co_t))
    # log p(Y = 1, | G_i = at, \theta)
    log_prob_y1_given_at = X_z1_w1 * beta_at - log1p.(exp.(X_z1_w1 * beta_at))
    # log p(Y = 0, | G_i = co, \theta)
    log_prob_y0_given_co_t = -log1p.(exp.(X_z1_w1 * beta_co_t))
    # log p(Y = 0, | G_i = at, \theta)
    log_prob_y0_given_at = -log1p.(exp.(X_z1_w1 * beta_at))
    # log p(G_i = co | Y_i, \theta)
    logp = Y_z1_w1 .* (log_prob_co + log_prob_y1_given_co_t - log.(exp.(log_prob_co + log_prob_y1_given_co_t) .+ exp.(log_prob_at + log_prob_y1_given_at)))
    logp += (1 .- Y_z1_w1) .* (log_prob_co + log_prob_y0_given_co_t - log.(exp.(log_prob_co + log_prob_y0_given_co_t) .+ exp.(log_prob_at + log_prob_y0_given_at)))
    # deciding co or at
    threshold = log.(rand(N_z1_w1))
    G_z1_w1 = zeros(N_z1_w1)
    G_z1_w1[threshold .< logp] .= 2
    G[mask_z1_w1] = G_z1_w1

    return G
end

function W_obs_and_mis_sampler(model::BayesianIV, G)
    W_obs_and_mis = zeros(model.N, 2)
    
    W_obs_and_mis[G .== 0, 1] .= 1
    W_obs_and_mis[G .== 0, 2] .= 1
    
    W_obs_and_mis[G .== 1, 1] .= 0
    W_obs_and_mis[G .== 1, 2] .= 0

    W_obs_and_mis[G .== 2, 1] .= 0
    W_obs_and_mis[G .== 2, 2] .= 1
    
    return W_obs_and_mis
end

function Y_obs_and_mis_sampler(model::BayesianIV, G, beta_co_c, beta_co_t)
    Y_obs_and_mis = fill(NaN, model.N, 2)
    
    # pattern for at
    Y_obs_and_mis[G .== 0, 2] = model.Y[G .== 0]
    # pattern for nt
    Y_obs_and_mis[G .== 1, 1] = model.Y[G .== 1]
    
    mask_co_c = (G .== 2) .& (model.Z .== 0)
    N_co_c = sum(mask_co_c)
    mask_co_t = (G .== 2) .& (model.Z .== 1)
    N_co_t = sum(mask_co_t)

    # pattern for co, c
    if N_co_c > 0
        X_co_c = model.X[mask_co_c, :]
        log_prob_y1_given_co_c = X_co_c * beta_co_c - log1p.(exp.(X_co_c * beta_co_c))
        Y_obs_and_mis[mask_co_c, 1] = model.Y[mask_co_c]
        Y_obs_and_mis[mask_co_c, 2] = log.(rand(N_co_c)) .< log_prob_y1_given_co_c
    end

    # pattern for co, t
    if N_co_t > 0
        X_co_t = model.X[mask_co_t, :]
        log_prob_y1_given_co_t = X_co_t * beta_co_t - log1p.(exp.(X_co_t * beta_co_t))
        Y_obs_and_mis[mask_co_t, 1] = log.(rand(N_co_t)) .< log_prob_y1_given_co_t
        Y_obs_and_mis[mask_co_t, 2] = model.Y[mask_co_t]
    end

    return Y_obs_and_mis
end

function tau_late_sampler(Y_mis_and_obs, G)
    mask_co = G .== 2
    N_co = sum(mask_co)

    if N_co > 0
        tau_late = sum(Y_mis_and_obs[mask_co, 2] - Y_mis_and_obs[mask_co, 1]) / N_co
        return tau_late
    else
        return NaN
    end
end

function gamma_sampler(model::BayesianIV, G, gamma_at, gamma_nt, step_size, num_step)
    r_init = randn(2 * model.dim)
    r = r_init - step_size / 2 * vcat(
        d_nlp_d_gamma_at(model, G, gamma_at, gamma_nt),
        d_nlp_d_gamma_nt(model, G, gamma_at, gamma_nt)
    )
    gamma_at_new = gamma_at + step_size * r[1:model.dim]
    gamma_nt_new = gamma_nt + step_size * r[model.dim+1:end]

    for _ in 1:num_step-1
        r -= step_size * vcat(
            d_nlp_d_gamma_at(model, G, gamma_at_new, gamma_nt_new),
            d_nlp_d_gamma_nt(model, G, gamma_at_new, gamma_nt_new)
        )
        gamma_at_new += step_size * r[1:model.dim]
        gamma_nt_new += step_size * r[model.dim+1:end]
    end

    r -= step_size / 2 * vcat(
        d_nlp_d_gamma_at(model, G, gamma_at_new, gamma_nt_new),
        d_nlp_d_gamma_nt(model, G, gamma_at_new, gamma_nt_new)
    )

    # deciding accept gamma_at and gamma_nt or not
    lp_old = log_posterior_gamma(model, G, gamma_at, gamma_nt)
    lp_new = log_posterior_gamma(model, G, gamma_at_new, gamma_nt_new)
    threshold = log(rand())
    if threshold < lp_new - sum(r[1:2*model.dim].^2) / 2 - lp_old + sum(r_init[1:2*model.dim].^2) / 2
        gamma_at_prop = gamma_at_new
        gamma_nt_prop = gamma_nt_new
    else
        gamma_at_prop = gamma_at
        gamma_nt_prop = gamma_nt
    end

    return gamma_at_prop, gamma_nt_prop
end

function beta_sampler(model::BayesianIV, G, beta_at, beta_nt, beta_co_c, beta_co_t, step_size, num_step)
    r_init = randn(4 * model.dim)
    r = r_init - step_size / 2 * vcat(
        d_nlp_d_beta_at(model, G, beta_at),
        d_nlp_d_beta_nt(model, G, beta_nt),
        d_nlp_d_beta_co_c(model, G, beta_co_c),
        d_nlp_d_beta_co_t(model, G, beta_co_t)
    )
    beta_at_new = beta_at + step_size * r[1:model.dim]
    beta_nt_new = beta_nt + step_size * r[model.dim+1:2*model.dim]
    beta_co_c_new = beta_co_c + step_size * r[2*model.dim+1:3*model.dim]
    beta_co_t_new = beta_co_t + step_size * r[3*model.dim+1:end]

    for _ in 1:num_step-1
        r -= step_size * vcat(
            d_nlp_d_beta_at(model, G, beta_at_new),
            d_nlp_d_beta_nt(model, G, beta_nt_new),
            d_nlp_d_beta_co_c(model, G, beta_co_c_new),
            d_nlp_d_beta_co_t(model, G, beta_co_t_new)
        )
        beta_at_new += step_size * r[1:model.dim]
        beta_nt_new += step_size * r[model.dim+1:2*model.dim]
        beta_co_c_new += step_size * r[2*model.dim+1:3*model.dim]
        beta_co_t_new += step_size * r[3*model.dim+1:end]
    end

    r -= step_size / 2 * vcat(
        d_nlp_d_beta_at(model, G, beta_at_new),
        d_nlp_d_beta_nt(model, G, beta_nt_new),
        d_nlp_d_beta_co_c(model, G, beta_co_c_new),
        d_nlp_d_beta_co_t(model, G, beta_co_t_new)
    )

    # Acceptance logic for beta_at
    lp_old = log_posterior_beta_at(model, G, beta_at)
    lp_new = log_posterior_beta_at(model, G, beta_at_new)
    threshold = log(rand())
    beta_at_prop = lp_new - sum(r[1:model.dim].^2) / 2 - lp_old + sum(r_init[1:model.dim].^2) / 2 > threshold ? beta_at_new : beta_at

    # Acceptance logic for beta_nt
    lp_old = log_posterior_beta_nt(model, G, beta_nt)
    lp_new = log_posterior_beta_nt(model, G, beta_nt_new)
    threshold = log(rand())
    beta_nt_prop = lp_new - sum(r[model.dim+1:2*model.dim].^2) / 2 - lp_old + sum(r_init[model.dim+1:2*model.dim].^2) / 2 > threshold ? beta_nt_new : beta_nt

    # Acceptance logic for beta_co_c
    lp_old = log_posterior_beta_co_c(model, G, beta_co_c)
    lp_new = log_posterior_beta_co_c(model, G, beta_co_c_new)
    threshold = log(rand())
    beta_co_c_prop = lp_new - sum(r[2*model.dim+1:3*model.dim].^2) / 2 - lp_old + sum(r_init[2*model.dim+1:3*model.dim].^2) / 2 > threshold ? beta_co_c_new : beta_co_c

    # Acceptance logic for beta_co_t
    lp_old = log_posterior_beta_co_t(model, G, beta_co_t)
    lp_new = log_posterior_beta_co_t(model, G, beta_co_t_new)
    threshold = log(rand())
    beta_co_t_prop = lp_new - sum(r[3*model.dim+1:end].^2) / 2 - lp_old + sum(r_init[3*model.dim+1:end].^2) / 2 > threshold ? beta_co_t_new : beta_co_t

    return beta_at_prop, beta_nt_prop, beta_co_c_prop, beta_co_t_prop
end

function d_nlp_d_gamma_at(model::BayesianIV, G, gamma_at, gamma_nt)
    d_log_prior = (model.N_a / 12 / model.N) * (sum(model.X, dims=1)[:] - 3 * sum(model.X .* (exp.(model.X * gamma_at) ./ (1 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))), dims=1)[:])
    if sum(G .== 1) > 0
        X_at = model.X[G .== 0, :]
        d_log_likelihood = sum(X_at, dims=1)[:] - sum(model.X .* (exp.(model.X * gamma_nt) ./ (1 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))), dims=1)[:]
    else
        d_log_likelihood = -sum(model.X .* (exp.(model.X * gamma_at) ./ (1 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))), dims=1)[:]
    end
    return -d_log_likelihood - d_log_prior
end

function d_nlp_d_gamma_nt(model::BayesianIV, G, gamma_at, gamma_nt)
    d_log_prior = (model.N_a / 12 / model.N) * (sum(model.X, dims=1)[:] - 3 * sum(model.X .* (exp.(model.X * gamma_nt) ./ (1 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))), dims=1)[:])
    if sum(G .== 1) > 0
        X_nt = model.X[G .== 1, :]
        d_log_likelihood = sum(X_nt, dims=1)[:] - sum(model.X .* (exp.(model.X * gamma_nt) ./ (1 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))), dims=1)[:]
    else
        d_log_likelihood = -sum(model.X .* (exp.(model.X * gamma_nt) ./ (1 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))), dims=1)[:]
    end
    return -d_log_likelihood - d_log_prior
end

function log_posterior_gamma(model::BayesianIV, G, gamma_at, gamma_nt)
    log_posterior = 0.0
    if sum(G .== 0) > 0
        X_at = model.X[G .== 0, :]
        log_posterior += sum(X_at * gamma_at)
    end
    if sum(G .== 1) > 0
        X_nt = model.X[G .== 1, :]
        log_posterior += sum(X_nt * gamma_nt)
    end
    log_posterior += -sum(log1p.(exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt)))
    log_posterior += (model.N_a / 12 / model.N) * sum(model.X * gamma_at)
    log_posterior += (model.N_a / 12 / model.N) * sum(model.X * gamma_nt)
    log_posterior += (model.N_a / 12 / model.N) * -3 * sum(log1p.(exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt)))
    return log_posterior
end

function d_nlp_d_beta_at(model::BayesianIV, G, beta_nt)
    d_log_prior = (2 * model.N_a / 12 / model.N) * (sum(model.X, dims=1) .- 2 * sum(model.X .* (exp.(model.X * beta_at) ./ (1 .+ exp.(model.X * beta_at))), dims=1))
    if sum(G .== 0) > 0
        X_at = model.X[G .== 0, :]
        Y_at = model.Y[G .== 0]
        d_log_likelihood = sum(Y_at .* X_at, dims=1) .- sum(X_at .* (exp.(X_at * beta_at) ./ (1 .+ exp.(X_at * beta_at))), dims=1)
        return - d_log_likelihood' - d_log_prior'
    else
        return - d_log_prior'
    end
end

function log_posterior_beta_at(model::BayesianIV, G, beta_at)
    log_posterior = 0.0
    if sum(G .== 0) > 0
        X_at = model.X[G .== 0, :]
        Y_at = model.Y[G .== 0]
        log_posterior += sum(Y_at .* (X_at * beta_at)) - sum(log1p.(exp.(X_at * beta_at)))
    end
    log_posterior += (2 * model.N_a / 12 / model.N) * (sum(model.X * beta_at) - 2 * sum(log1p.(exp.(model.X * beta_at))))
    return log_posterior
end

function d_nlp_d_beta_nt(model::BayesianIV, G, beta_nt)
    d_log_prior = (2 * model.N_a / 12 / model.N) * (sum(model.X, dims=1) .- 2 * sum(model.X .* (exp.(model.X * beta_nt) ./ (1 .+ exp.(model.X * beta_nt))), dims=1))
    if sum(G .== 1) > 0
        X_nt = model.X[G .== 1, :]
        Y_nt = model.Y[G .== 1]
        d_log_likelihood = sum(Y_nt .* X_nt, dims=1) .- sum(X_nt .* (exp.(X_nt * beta_nt) ./ (1 .+ exp.(X_nt * beta_nt))), dims=1)
        return - d_log_likelihood' - d_log_prior'
    else
        return - d_log_prior'
    end
end

function log_posterior_beta_nt(model::BayesianIV, G, beta_nt)
    log_posterior = 0.0
    if sum(G .== 1) > 0
        X_nt = model.X[G .== 1, :]
        Y_nt = model.Y[G .== 1]
        log_posterior += sum(Y_nt .* (X_nt * beta_nt)) - sum(log1p.(exp.(X_nt * beta_nt)))
    end
    log_posterior += (2 * model.N_a / 12 / model.N) * (sum(model.X * beta_nt) - 2 * sum(log1p.(exp.(model.X * beta_nt))))
    return log_posterior
end

function d_nlp_d_beta_co_c(model::BayesianIV, G, beta_co_c)
    d_log_prior = (model.N_a / 12 / model.N) * (sum(model.X, dims=1) .- 2 * sum(model.X .* (exp.(model.X * beta_co_c) ./ (1 .+ exp.(model.X * beta_co_c))), dims=1))
    if sum((G .== 2) .& (model.Z .== 0)) > 0
        mask_co_c = (G .== 2) .& (model.Z .== 0)
        X_co_c = model.X[mask_co_c, :]
        Y_co_c = model.Y[mask_co_c]
        d_log_likelihood = sum(Y_co_c .* X_co_c, dims=1) .- sum(X_co_c .* (exp.(X_co_c * beta_co_c) ./ (1 .+ exp.(X_co_c * beta_co_c))), dims=1)
        return - d_log_likelihood' - d_log_prior'
    else
        return - d_log_prior'
    end
end

function log_posterior_beta_co_c(model::BayesianIV, G, beta_co_c)
    log_posterior = 0.0
    if sum((G .== 2) .& (model.Z .== 0)) > 0
        mask_co_c = (G .== 2) .& (model.Z .== 0)
        X_co_c = model.X[mask_co_c, :]
        Y_co_c = model.Y[mask_co_c]
        log_posterior += sum(Y_co_c .* (X_co_c * beta_co_c)) - sum(log1p.(exp.(X_co_c * beta_co_c)))
    end
    log_posterior += (model.N_a / 12 / model.N) * (sum(model.X * beta_co_c) - 2 * sum(log1p.(exp.(model.X * beta_co_c))))
    return log_posterior
end

function d_nlp_d_beta_co_t(model::BayesianIV, G, beta_co_t)
    d_log_prior = (model.N_a / 12 / model.N) * (sum(model.X, dims=1) .- 2 * sum(model.X .* (exp.(model.X * beta_co_t) ./ (1 .+ exp.(model.X * beta_co_t))), dims=1))
    if sum((G .== 2) .& (model.Z .== 1)) > 0
        mask_co_t = (G .== 2) .& (model.Z .== 1)
        X_co_t = model.X[mask_co_t, :]
        Y_co_t = model.Y[mask_co_t]
        d_log_likelihood = sum(Y_co_t .* X_co_t, dims=1) .- sum(X_co_t .* (exp.(X_co_t * beta_co_t) ./ (1 .+ exp.(X_co_t * beta_co_t))), dims=1)
        return - d_log_likelihood' - d_log_prior'
    else
        return - d_log_prior'
    end
end

function log_posterior_beta_co_t(model::BayesianIV, G, beta_co_t)
    log_posterior = 0.0
    if sum((G .== 2) .& (model.Z .== 1)) > 0
        mask_co_t = (G .== 2) .& (model.Z .== 1)
        X_co_t = model.X[mask_co_t, :]
        Y_co_t = model.Y[mask_co_t]
        log_posterior += sum(Y_co_t .* (X_co_t * beta_co_t)) - sum(log1p.(exp.(X_co_t * beta_co_t)))
    end
    log_posterior += (model.N_a / 12 / model.N) * (sum(model.X * beta_co_t) - 2 * sum(log1p.(exp.(model.X * beta_co_t))))
    return log_posterior
end

function log_posterior(model::BayesianIV, G, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t)
    lp = 0.0

    # log p(G|\theta)
    prob_co = 1.0 ./ (1.0 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))
    prob_nt = exp.(model.X * gamma_nt) ./ (1.0 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))
    prob_at = exp.(model.X * gamma_at) ./ (1.0 .+ exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt))

    lp += sum(log.(prob_at[G .== 0]))
    lp += sum(log.(prob_nt[G .== 1]))
    lp += sum(log.(prob_co[G .== 2]))

    # log p(Y|G, \theta)
    # pattern for at
    X_at = model.X[G .== 0, :]
    Y_at = model.Y[G .== 0]
    lp += sum(Y_at .* (X_at * beta_at) - log1p.(exp.(X_at * beta_at)))

    # pattern for nt
    X_nt = model.X[G .== 1, :]
    Y_nt = model.Y[G .== 1]
    lp += sum(Y_nt .* (X_nt * beta_nt) - log1p.(exp.(X_nt * beta_nt)))

    # pattern for co_c
    mask_co_c = (G .== 2) .& (model.Z .== 0)
    X_co_c = model.X[mask_co_c, :]
    Y_co_c = model.Y[mask_co_c]
    lp += sum(Y_co_c .* (X_co_c * beta_co_c) - log1p.(exp.(X_co_c * beta_co_c)))

    # pattern for co_t
    mask_co_t = (G .== 2) .& (model.Z .== 1)
    X_co_t = model.X[mask_co_t, :]
    Y_co_t = model.Y[mask_co_t]
    lp += sum(Y_co_t .* (X_co_t * beta_co_t) - log1p.(exp.(X_co_t * beta_co_t)))

    # log p(\theta)
    lp += (model.N_a / 12 / model.N) * sum(model.X * gamma_at)
    lp += (model.N_a / 12 / model.N) * sum(model.X * gamma_nt)
    lp += -3 * (model.N_a / 12 / model.N) * sum(log1p.(exp.(model.X * gamma_at) .+ exp.(model.X * gamma_nt)))
    lp += (2 * model.N_a / 12 / model.N) * (sum(model.X * beta_at) - 2 * sum(log1p.(exp.(model.X * beta_at))))
    lp += (2 * model.N_a / 12 / model.N) * (sum(model.X * beta_nt) - 2 * sum(log1p.(exp.(model.X * beta_nt))))
    lp += (model.N_a / 12 / model.N) * (sum(model.X * beta_co_c) - 2 * sum(log1p.(exp.(model.X * beta_co_c))))
    lp += (model.N_a / 12 / model.N) * (sum(model.X * beta_co_t) - 2 * sum(log1p.(exp.(model.X * beta_co_t))))

    return lp
end

function sampling(
    model::BayesianIV,
    num_samples;
    thinning=1,
    burn_in=0,
    step_size=Dict("gamma" => 1e-3, "beta" => 1e-4),
    num_step=Dict("gamma" => 1, "beta" => 1),
    gamma_at_init=nothing,
    gamma_nt_init=nothing,
    beta_at_init=nothing,
    beta_nt_init=nothing,
    beta_co_c_init=nothing,
    beta_co_t_init=nothing
)
    gamma_at = isnothing(gamma_at_init) ? randn(model.dim) : gamma_at_init
    gamma_nt = isnothing(gamma_nt_init) ? randn(model.dim) : gamma_nt_init
    beta_at = isnothing(beta_at_init) ? randn(model.dim) : beta_at_init
    beta_nt = isnothing(beta_nt_init) ? randn(model.dim) : beta_nt_init
    beta_co_c = isnothing(beta_co_c_init) ? randn(model.dim) : beta_co_c_init
    beta_co_t = isnothing(beta_co_t_init) ? randn(model.dim) : beta_co_t_init

    G_samples = zeros(num_samples, model.N)
    gamma_at_samples = zeros(num_samples, model.dim)
    gamma_nt_samples = zeros(num_samples, model.dim)
    beta_at_samples = zeros(num_samples, model.dim)
    beta_nt_samples = zeros(num_samples, model.dim)
    beta_co_c_samples = zeros(num_samples, model.dim)
    beta_co_t_samples = zeros(num_samples, model.dim)
    lp_list = zeros(num_samples)

    @showprogress for i in 1:burn_in+num_samples
        for _ in 1:thinning
            G = G_sampler(model, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t)
            gamma_at, gamma_nt = gamma_sampler(model, G, gamma_at, gamma_nt, step_size["gamma"], num_step["gamma"])
            beta_at, beta_nt, beta_co_c, beta_co_t = beta_sampler(model, G, beta_at, beta_nt, beta_co_c, beta_co_t, step_size["beta"], num_step["beta"])
        end

        if i > burn_in
            G_samples[i-burn_in, :] = G
            gamma_at_samples[i-burn_in, :] = gamma_at
            gamma_nt_samples[i-burn_in, :] = gamma_nt
            beta_at_samples[i-burn_in, :] = beta_at
            beta_nt_samples[i-burn_in, :] = beta_nt
            beta_co_c_samples[i-burn_in, :] = beta_co_c
            beta_co_t_samples[i-burn_in, :] = beta_co_t
            lp_list[i-burn_in] = log_posterior(model, G, gamma_at, gamma_nt, beta_at, beta_nt, beta_co_c, beta_co_t)
        end
    end

    W_obs_and_mis_samples = [W_obs_and_mis_sampler(model, G_samples[i, :]) for i in 1:num_samples]
    Y_obs_and_mis_samples = [Y_obs_and_mis_sampler(model, G_samples[i, :], beta_co_c_samples[i, :], beta_co_t_samples[i, :]) for i in 1:num_samples]
    tau_late_samples = [tau_late_sampler(Y_obs_and_mis_samples[i], G_samples[i, :]) for i in 1:num_samples]

    return Dict(
        "G" => G_samples,
        "gamma_at" => gamma_at_samples,
        "gamma_nt" => gamma_nt_samples,
        "beta_at" => beta_at_samples,
        "beta_nt" => beta_nt_samples,
        "beta_co_c" => beta_co_c_samples,
        "beta_co_t" => beta_co_t_samples,
        "W_obs_and_mis" => W_obs_and_mis_samples,
        "Y_obs_and_mis" => Y_obs_and_mis_samples,
        "tau_late" => tau_late_samples,
        "lp" => lp_list
    )
end