
function gme_detection(ρ::AbstractMatrix{T}, dims::NTuple{N, Int}; measure::String = "2-norm", fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {T <: Number, N}
    LA.ishermitian(ρ) || throw(ArgumentError("State needs to be Hermitian"))
    N == 1 && throw(ArgumentError("At least two subsystems are required."))
    N == 2 && return entanglement_detection(ρ, dims; measure, fw_algorithm, kwargs...)
    return multipartite_entanglement_detection(ρ, dims, 2; measure, fw_algorithm, kwargs...)
end

function multipartite_entanglement_detection(ρ::AbstractMatrix{T}, dims::NTuple{N, Int}, k::Int; measure::String = "2-norm", fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {T <: Number, N}
    k > N && throw(ArgumentError("The number of entangled parties `k` should be smaller than number of states' parties."))
    k == N && return entanglement_detection(ρ, dims; measure, fw_algorithm, kwargs...)
    lmo = KSeparableLMO(T, dims, k; matrix_basis = _gellmann(T, dims), kwargs...)
    C = correlation_tensor(ρ, dims, _gellmann(T, dims))
    return separable_distance(C, lmo; measure, fw_algorithm, kwargs...)
end

"""
    entanglement_detection(ρ::AbstractMatrix{T}, dims::NTuple{N, Int}; measure, algorithm, kwargs...) where {T <: Number, N}

Given a quantum state `ρ` with subsystem dimensions as `dims`, make a judgment about whether it is entangled or separable.
If the argument `dims` is omitted equally-sized subsystems are assumed, which is solving on the symmetry bipartite separable space.

Returns a named tuple `(ent, atoms, witness)` with:
- `ent` true/false/nothing, the judgment as entangled/separable/can't tell
- `witness` the witness for the entangled `ρ`, `nothing` for separable `ρ`
- `decompose` a tuple with weights and pure separable states, the best separable decomposition for separable `ρ` or for the closest separable state of entangled `ρ`
"""
function entanglement_detection(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); measure::String = "2-norm", fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}
    LA.ishermitian(ρ) || throw(ArgumentError("State needs to be Hermitian"))
    N == 1 && throw(ArgumentError("At least two subsystems are required."))
    σ, _..., active_set, lmo = separable_distance(ρ, dims, lmo; measure, fw_algorithm, kwargs...) #TODO return atoms
    witness = entanglement_witness(ρ, σ, dims; lmo, kwargs...)
    if real(LA.dot(witness.W, ρ)) < 0
        return (ent = true, witness = witness, decompose = active_set)
    else
        if separable_ball_criterion(ρ, σ, dims; lmo, kwargs...)
            return (ent = false, witness = witness, decompose = active_set)
        else
            return (ent = nothing, witness = witness, decompose = active_set)
        end
    end
end
export entanglement_detection

"""
    separable_ball_criterion(ρ, dims; kwargs...)

Given a quantum state `ρ` with subsystem dimensions as `dims`, define a quantum state
```
ρ' = (1+t)/t * ρ - 1/t * σ, s.t. ρ = t/(1-t) * ρ' + 1/(1-t) * σ.
```
where `σ` is the closest separable pure state.
By confirming `ρ'` inside a separable ball to confirm the separability of `ρ`.

Returns a named tuple `(sep, atoms, witness)` with:
- `sep` true/false, the judgment of separability
- `p` = t/(1-t), the weight for `ρ'`
"""
function separable_ball_criterion(ρ::AbstractMatrix{T}, dims::NTuple{N, Int}; kwargs...) where {T <: Number, N}
    σ, _..., lmo = separable_distance(ρ, dims; kwargs...)
    return separable_ball_criterion(ρ, σ, dims; lmo, kwargs...)
end
export separable_ball_criterion

"""
    separable_ball_criterion(ρ, σ, dims; kwargs...)

Given a quantum state `ρ` and a separable state `σ`, define a quantum state
```
ρ' = (1+t)/t * ρ - 1/t * σ, s.t. ρ = t/(1-t) * ρ' + 1/(1-t) * σ.
```
By confirming ρ' inside a separable ball to confirm the separability of `ρ`.

Returns a named tuple `(sep, atoms, witness)` with:
- `sep` true/false, the judgment of separability
- `p` = t/(1-t), the weight for `ρ'`
"""
function separable_ball_criterion(ρ::AbstractMatrix{T}, σ::AbstractMatrix{T}, dims::NTuple{N, Int}; t_max = 200, t_th = 1e-4, kwargs...) where {T <: Number, N}
    R = real(T)
    if separable_ball_membership(ρ, dims; kwargs...)
        return true
    else
        t = R(t_max)
        while t > t_th
            if separable_ball_membership(_extend(ρ, σ, t), dims; kwargs...) #TODO more general method
                return true
            else
                t -= t_th
            end
        end
    end
    return false
end
_extend(ρ, σ, t) = (1 + t) / t * ρ - 1 / t * σ

"""
    separable_ball_membership(ρ::AbstractMatrix{T}, dims::NTuple{N, Int}; kwargs...)

Judges whether a given quantum state `ρ` in a separable ball with subsystem dimensions as `dims`.
"""
function separable_ball_membership(ρ::AbstractMatrix{T}, dims::NTuple{N, Int}; kwargs...) where {T <: Number, N}
    N == 1 && throw(ArgumentError("At least two subsystems are required."))
    r = sqrt(real(LA.tr((ρ - Matrix(LA.I, size(ρ)) / size(ρ, 1))^2)))
    return r < separable_ball_radius(real(T), dims)
end

"""
    separable_ball_radius(::Type{T}, dims::NTuple{N, Int})

Lower bound of the radius of a separable ball, centered around the maximally mixed state.

Reference:
- [N-qubits](https://arxiv.org/abs/quant-ph/0601201)
- [N-qudits](https://arxiv.org/abs/quant-ph/0409095)
- [general case](https://doi.org/10.1103/PhysRevA.68.042312)
"""
function separable_ball_radius(::Type{T}, dims::NTuple{N, Int}) where {T <: Number, N}
    N == 1 && throw(ArgumentError("At least two subsystems are required."))
    if length(unique(dims)) ≤ 1 # symmetry
        if dims[1] == 2 && N ≥ 3 # N-qubits (https://arxiv.org/abs/quant-ph/0601201)
            return sqrt(54 / 17) * 6^(-N / 2)
        else # N-qudits (https://arxiv.org/abs/quant-ph/0409095)
            return 1 / sqrt((2 * dims[1] - 1)^(N - 2) * (dims[1]^2 - 1) * dims[1]^N + dims[1]^N)
        end
    else # asymmetry (https://doi.org/10.1103/PhysRevA.68.042312)
        return 1 / (2^(N / 2 - 1) * prod(dims))
    end
end
separable_ball_radius(dims) = separable_ball_radius(Float64, dims)
export separable_ball_radius
