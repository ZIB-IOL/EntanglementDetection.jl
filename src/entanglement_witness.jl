"""
    entanglement_witness(ρ::AbstractMatrix{T}; dims, kwargs...)

Given a quantum state `ρ`, find the best witness that detect the entanglement of the state, defined by
```
W = (σ-ρ + Tr[σ(ρ-σ)]*I)/||ρ-σ||
```
which satisfies ∀σ ∈ SEP, Tr(Wσ) ≥ 0, and ∃ρ, Tr(Wρ) < 0.

Returns a tuple `(W, σ, α)` with:
- `W` the witness operator
- `σ` the closest pure separable state
- `α` = Tr[σ(ρ-σ)], the overlap
"""
function entanglement_witness(ρ::AbstractMatrix{T}, dims::NTuple{N, Int}; kwargs...) where {T <: Number, N}
    LA.ishermitian(ρ) || throw(ArgumentError("State needs to be Hermitian"))
    N == 1 && throw(ArgumentError("At least two subsystems are required."))
    σ, _..., lmo = separable_distance(ρ, dims; measure, algorithm, kwargs...)
    return entanglement_witness(ρ, σ, dims; lmo, kwargs...)
end
export entanglement_witness

"""
    entanglement_witness(ρ::AbstractMatrix, σ::AbstractMatrix; dims, kwargs...)

Given a direction from a state `σ` in the separable space to a entangled state `ρ`, find the best witness that detect the entanglement of the state along this direction:
```
W = (σ-ρ + Tr[σ(ρ-σ)]*I)/||ρ-σ||
```
which satisfies ∀σ ∈ SEP, Tr(Wσ) ≥ 0, and ∃ρ, Tr(Wρ) < 0.

Returns a named tuple `(W, α, ε, σ, ϕ)` with:
- `W` the witness operator
- `α` = Tr[ϕ(ρ-σ)]/||ρ-σ||, the overlap
- `ε` the ε-net, normalized by ||ρ-σ||
- `β` = Tr[σ(ρ-σ)]/||ρ-σ||, the overlap
- `σ` decide the direction from `σ` to `ρ`
- `ϕ` the closest pure separable state
"""
function entanglement_witness(ρ::AbstractMatrix{CT}, σ::AbstractMatrix{CT}, dims::NTuple{N, Int}; lmo = nothing, max_length = 10^6, min_η = 1 - 1e-3, kwargs...) where {CT <: Number, N}
    if lmo === nothing
        lmo = EnumeratingSeparableLMO(real(CT), dims; max_length, min_η, kwargs...)
    end
    ∇ = σ - ρ
    distance = LA.norm(∇)
    dir = correlation_tensor(∇, lmo.dims, lmo.matrix_basis)
    ϕ = density_matrix(FrankWolfe.compute_extreme_point(lmo, dir), lmo.dims, lmo.matrix_basis) # Enumerate to achieve the closest pure separable state
    α = real(LA.dot(ϕ, ∇))
    η = radius_inner(lmo)
    ε = (1 - η) * distance
    W = ∇ - (α - ε) * Matrix(LA.I, size(ρ))
    W ./= distance
    α /= distance
    ε /= distance
    return (W = W, α = α, ε = ε, σ = σ, ϕ = ϕ, distance = distance, η = η)
end

function entanglement_witness(C::Array{T, N}, x::Array{T, N}, dims::NTuple{N, Int}; lmo = nothing, max_length = 10^6, min_η = 1 - 1e-3, kwargs...) where {T <: Real, N}
    if lmo === nothing
        lmo = EnumeratingSeparableLMO(T, dims; max_length, min_η, kwargs...)
    end
    Id = similar(C)
    Id .= T(0)
    Id[1] = T(1) * 2^N
    dir = x - C
    distance = LA.norm(dir) / sqrt(2^N)
    y = FrankWolfe.compute_extreme_point(lmo, dir) # Enumerate to achieve the closest pure separable state
    α = LA.dot(y, dir) / 2^N
    η = radius_inner(lmo)
    ε = (1 - η) * distance
    W = dir - (α - ε) * Id
    W ./= distance
    α /= distance
    ε /= distance
    return (W = W, α = α, ε = ε, σ = x, ϕ = y, distance = distance, η = η)
end
