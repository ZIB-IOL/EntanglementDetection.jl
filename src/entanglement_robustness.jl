"""
    entanglement_robustness(ρ_p::Function, p_list::Array{Vector{T}}, dims::NTuple{N, Int}; monotone, robust_monitor, kwargs...)

Decide the entanglement and separable regions for a family of quantum state `ρ_p`.
    
Inputs:
- `ρ_p` the function for a family of quantum states
- `p_list` the parameter region for `ρ_p(p)` 
- `monotone` the monotony of the function if with only one parameter (default: true)
- `robust_monitor` the monitor for the calculation of the robustness problem

Returns a named tuple `(ent_range, nan_range, sep_range)` with:
- `ent_range` the parameter range for `p` such that `ρ_p(p)` is entangled.
- `nan_range` the parameter range for `p` such that `ρ_p(p)` can not be decided.
- `sep_range` the parameter range for `p` such that `ρ_p(p)` is separable.
"""
function entanglement_robustness(ρ_p::Function, p_list::Array{Vector{T}}, dims::NTuple{N, Int}; monotone = true, robust_monitor = true, kwargs...) where {T <: Real, N}
    ent_range = Vector{T}[]
    sep_range = Vector{T}[]
    nan_range = Vector{T}[]
    active_set = nothing
    p = p_list[1]
    while p in p_list
        ent, witness, active_set = entanglement_detection(ρ_p(p), dims; active_set, kwargs...)
        if isnothing(ent)
            push!(nan_range, p)
            p_list = setdiff(p_list, nan_range)
        elseif ent
            push!(ent_range, p)
            for p2 in p_list
                if real(LA.dot(witness.W, ρ_p(p2))) < 0
                    push!(ent_range, p2)
                end
            end
            p_list = setdiff(p_list, ent_range)
        else
            if monotone
                sep_range = p_list
                robust_monitor&&_monitor(ent_range, nan_range, sep_range)
                break
            else
                push!(sep_range, p)
                p_list = setdiff(p_list, sep_range)
            end
        end
        robust_monitor&&_monitor(ent_range, nan_range, sep_range)
        if isempty(p_list)
            break
        end
        p = p_list[1]
    end
    println()
    println()
    println()
    return (ent_range = ent_range, nan_range = nan_range, sep_range = sep_range,)
end
export entanglement_robustness

function _monitor(range1::Vector{Vector{T}}, range2::Vector{Vector{T}}, range3::Vector{Vector{T}}) where T
    if isempty(range1)
        print("\rEnt: []")
    else
        print("\rEnt: ", range1[end], "  ")
    end
    if isempty(range2)
        print("\nGap: []")
    else
        print("\nGap: ", range2[end], "  ")
    end
    if isempty(range3)
        print("\nSep: []")
    else
        print("\nSep: [", range3[end], "  ")
    end
    print("\e[F\e[F")
end

function linear_noise_robustness(ρ::AbstractMatrix{CT}, σ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); noise_atol::Real = 1e-3, kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}
    LA.eigmin(Ket.partial_transpose(σ, [1], collect(dims))) < 0 && @warn "The noise is NPT, not separable. The robust noise level could be not completed."
    @time res1 = linear_noise_robustness_ent_bound(ρ, σ, dims, lmo; noise_atol, kwargs...)
    @time res2 = linear_noise_robustness_sep_bound(ρ, σ, dims, lmo; active_set = res1.active_set, noise_level = res1.ent_bound, kwargs...)
    return (ent_bound = res1.ent_bound, sep_bound = res2.sep_bound)
end
export linear_noise_robustness

function linear_noise_robustness_ent_bound(ρ::AbstractMatrix{CT}, σ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); noise_atol::Real = 1e-3, kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}
    C = correlation_tensor(ρ, lmo.dims, lmo.matrix_basis)
    noise = correlation_tensor(σ, lmo.dims, lmo.matrix_basis)
    x, v, primal, noise_level, active_set, lmo = separable_distance(C, lmo; noise_mixture = true, noise, noise_atol, kwargs...)
    lmo_enum = EnumeratingSeparableLMO(real(CT), lmo.dims; matrix_basis = lmo.matrix_basis, kwargs...)
    ρ_noisy = (1 - noise_level) * ρ + noise_level * σ
    dir = correlation_tensor(σ - ρ_noisy, lmo.dims, lmo.matrix_basis)
    ϕ = density_matrix(FrankWolfe.compute_extreme_point(lmo_enum, dir), lmo_enum.dims, lmo_enum.matrix_basis)
    # witness = entanglement_witness(ρ_noisy, σ, dims)
    if LA.norm(ρ_noisy- ϕ) > 1 - lmo_enum.approximations[1].η
        ent_bound = noise_level
    else
        # ent_bound = - real(LA.dot(witness.W, ρ)) / real(LA.dot(witness.W, σ - ρ))
        ent_bound = nothing
    end
    return (ent_bound = ent_bound, primal =primal, active_set = active_set)
end

function linear_noise_robustness_sep_bound(ρ::AbstractMatrix{CT}, σ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); noise_level::Real = 0.0, active_set, noise_atol::Real = 1e-3, kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}
    for sep_bound in noise_level:noise_atol:1.0
        rho = (1 - sep_bound) * ρ + sep_bound * σ
        sigma, _..., active_set, lmo = separable_distance(rho, dims, lmo; active_set, noise_mixture = false, kwargs...)
        if separable_ball_criterion(rho, sigma, dims)
            return (sep_bound = sep_bound, )
        end
    end
    return (sep_bound = 1.0, )
end

"""
function white_noise_robustness(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO; ent_proof, sep_search, noise_atol, kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}

Given a quantum state `ρ`, calculate the upper and lower bounds for the amount of white noise it can tolerate to be separable.

- `ent_proof`: `true`/`false` use the enumerate method to guarantee the entanglement
- `sep_search`: `true`/`false` use the linear search method to find a possible better separable bound
- `noise_atol`: the numerical accuracy for the noise level

Returns a tuple `(ent_bound, sep_bound)`.
"""
function white_noise_robustness(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); ent_proof = false, sep_search = false, noise_atol::Real = 1e-3, max_length = 10^6, min_η = 1 - 1e-3, kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}
    σ = Matrix{CT}(LA.I, prod(dims), prod(dims))./prod(dims)
    C = correlation_tensor(ρ, lmo.dims, lmo.matrix_basis)
    noise = correlation_tensor(σ, lmo.dims, lmo.matrix_basis)
    x, v, primal, noise_level, active_set, lmo = separable_distance(C, lmo; noise_mixture = true, noise, noise_atol, kwargs...)
    if ent_proof
        ρ_noisy = (1 - noise_level) * ρ + noise_level * σ
        witness = entanglement_witness(ρ_noisy, density_matrix(x, lmo.dims, lmo.matrix_basis), dims; max_length, min_η)
        distance = LA.norm(ρ_noisy - witness.ϕ)
        η = witness.η
        if real(LA.dot(witness.W, ρ_noisy)) < 0 # entanglement proof
            ent_bound = noise_level
        else
            ent_bound = - real(LA.dot(witness.W, ρ)) / real(LA.dot(witness.W, σ - ρ))
        end
    else
        distance = sqrt(primal/2^(N-1))
        η = 1.0
        ent_bound = noise_level
    end
    if sep_search
        sep_bound = 1.0
        for bound in noise_level:noise_atol:1.0
            rho = (1 - bound) * ρ + bound * σ
            sigma, _..., active_set, lmo = separable_distance(rho, dims, lmo; active_set, noise_mixture = false, kwargs...)
            if separable_ball_criterion(rho, sigma, dims)
                sep_bound = bound
                break
            end
        end
    else
        sep_bound = separable_bound_with_white_noise(dims, noise_level, sqrt(primal/2^(N-1)))
    end
    return (ent_bound = ent_bound, sep_bound = sep_bound, ent_proof = ent_proof, sep_search = sep_search, distance = distance, η = η)
end
export white_noise_robustness

"""
    separable_bound_with_white_noise(::Type{T}, dims::NTuple{N, Int}, ent_bound::T, distance::T) where {T <: Real, N}

Consider a quantum state mixed with white noise as ```ρ(p) = (1-p) * ρ + p * I/d``` for the separability problem. 

Given
- `dims` for the structure of the separable space
- `ent_bound` the noise level such that ρ(`ent_bound`) could be entangled (not necessary)
- `distance` = min_{σ ∈ SEP}||ρ(`ent_bound`) - σ||, the distance between the state and the separable space

Return the noise level such that ρ(`sep_bound`) must be separable.
"""
function separable_bound_with_white_noise(dims::NTuple{N, Int}, ent_bound::T, distance::T) where {T <: Real, N}
    r = separable_ball_radius(T, dims)
    ϵ = distance / r
    return T((ent_bound + ϵ)/(1 + ϵ))
end

"""
    entangled_bound_with_white_noise(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}; atol = 1e-4) where {CT <: Number, N}
Given a quantum state `ρ`, using PPT criterion calculate the upper bound for the amount of white noise it can tolerate such that
```ρ = (1-p) * ρ + p * I/d``` is entangled.
Note this is corresponding to the fully separability.
Returns the upper bound `p`.
"""
function entangled_bound_with_white_noise(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}; atol = 1e-4) where {CT <: Number, N}
    p = zero(real(CT))
    Id = Matrix{CT}(LA.I, prod(dims), prod(dims))./prod(dims)
    for partition in _partitions(1:N, 2)
        while LA.eigmin(Ket.partial_transpose((1-p)*ρ + p*Id, partition[1], collect(dims))) < 0 
            p += atol
        end
    end
    p -= atol
    return p
end

# #TODO not be used yet
# """
#     white_noise_robustness(ρ_t::AbstractMatrix{T}; dims, kwargs...)

# Given a quantum state `ρ_t`, calculate the upper bound for the amount of white noise it can tolerate such that

# ```
# ρ = (1-p) * ρ_t + p * I/d
# ```
# is separable.

# Returns a tuple `(t, res)` with:
# - `p` the upper bound
# - `res`
# """
# function white_noise_robustness(ρ_t::AbstractMatrix{CT}, dims::NTuple{N, Int}; kwargs...) where {CT <: Number, N}
#     x, _ = separable_distance(ρ_t, dims; kwargs...)
#     return white_noise_robustness(ρ_t, x, dims; kwargs...)
# end
# export white_noise_robustness

# """
#     white_noise_robustness(ρ_t::AbstractMatrix{T}, x; dims, kwargs...)

# Given a quantum state `ρ_t` and a separable state `x`, by defining a quantum state
# ```
# ρ_x = 1/t * ρ_t + I/d - 1/t * x
# ```
# and confirming ρ_x inside a separable ball to confirm the separability of
# ```
# ρ = (1-p) * ρ_t + p * I/d, where p = t/(1+t)
# ```

# Returns a tuple `(t, res)` with:
# - `p` the upper bound
# - `res`
# """
# function white_noise_robustness(ρ_t::AbstractMatrix{CT}, x::AbstractMatrix{CT}, dims::NTuple{N, Int}; robust_atol = 1e-3, kwargs...) where {CT <: Number, N}
#     R = real(CT)
#     p = one(R)
#     t = R((1 - robust_atol)/robust_atol)
#     #TODO fast to bound
#     while t > robust_atol / (1 - robust_atol)
#         if separable_ball_membership(_shrink(ρ_t, x, t), dims; kwargs...) #TODO more general method
#             p = R(t / (1 + t))
#         end
#         _monitor(t, p)
#         t -= (1+t)^2/(1/robust_atol + (1+t))
#     end
#     return (p = p,)
# end
# _shrink(ρ_t, x, t) = LA.Hermitian(1 / t * ρ_t + LA.I(size(ρ_t, 1)) / size(ρ_t, 1) - 1 / t * x)
# function _monitor(t::T, p::T) where {T <: Number}
#     # print("\r[t, p]: ", [t, p], "  ")
#     println("[t, p]: ", [t, p], "  ")
# end