# Ket does not normalise gellmann the same way we do, the first element of tensor should be treated differently
function _gellmann(::Type{CT}, dims::NTuple{N, Int}) where {CT <: Number, N}
    T = float(real(CT))
    matrix_basis = broadcast.(Matrix{Complex{T}}, Ket.gellmann.(Complex{T}, dims))
    for n in 1:N
        matrix_basis[n][1] .*= sqrt(T(2)) / sqrt(T(dims[n]))
    end
    return matrix_basis
end

# https://github.com/jmichel7/Combinat.jl
function _partitions(set::AbstractVector, k)
    res = Vector{Vector{eltype(set)}}[]
    if length(set) < k
        return res
    end
    if k == 1
        return [[collect(set)]]
    end
    for p in _partitions(set[1:end-1], k - 1)
        push!(res, vcat(p, [[set[end]]]))
    end
    for p in _partitions(set[1:end-1], k)
        for i in eachindex(p)
            u = copy(p)
            u[i] = vcat(u[i], [set[end]])
            push!(res, u)
        end
    end
    return res
end

function reconstruct_basis(basis::NTuple{N, Vector{MB}}, partition::Vector{Vector{Int}}) where {MB <: AbstractMatrix, N}
    matrix_basis = ntuple(i -> MB[], length(partition))
    for i in 1:length(partition)
        if length(partition[i]) == 1
            push!.((matrix_basis[i],), basis[partition[i]][1])
        else
            push!.((matrix_basis[i],), kron(basis[partition[i]]))
        end
    end
    return matrix_basis
end

function Base.kron(A::NTuple{N, Vector{MB}}) where {MB <: AbstractMatrix, N}
    dims = collect(length.(A))
    B = Vector{MB}(undef, prod(dims))
    vi = ones(Int, N)
    for i in 1:prod(dims)
        B[i] = kron((A[ni][di] for (ni, di) in enumerate(vi))...)
        _update_odometer!(vi, dims)
    end
    return B
end

function group_dims(tensor::AbstractArray{T, N1}, new_dims::NTuple{N2, Int}, partition::Vector{Vector{Int}}) where {T <: Real, N1, N2}
    return reshape(PermutedDimsArray(tensor, vcat(partition...)), new_dims.^2)
end

function ungroup_dims(tensor::AbstractArray{T, N1}, dims::NTuple{N2, Int}, previous_partition::Vector{Vector{Int}}) where {T <: Real, N1, N2}
    return PermutedDimsArray(reshape(tensor, dims.^2), sortperm(vcat(previous_partition...)))
end

function ungroup_dims(ps::PureState{T, N1}, dims::NTuple{N2, Int}, previous_partition::Vector{Vector{Int}}) where {T <: Real, N1, N2}
    return PermutedDimsArray(reshape(kron([ps.tensors[i] for i in N1:-1:1]...), dims.^2), sortperm(vcat(previous_partition...)))
end

"""
    correlation_tensor(ρ::Matrix{T}, dims::NTuple{N, Int})

Convert a dimension-(a)symmetry density matrix `ρ` to a correlation tensor Array{T, N}, with subspace dimensions `dims`.
"""
function correlation_tensor(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, matrix_basis = _gellmann(CT, dims)) where {CT <: Number, N}
    @assert size(ρ) == (prod(dims), prod(dims)) "Density matrix size is not compatible with the given dimensions."
    T = float(real(CT))
    C = Array{T, N}(undef, dims .^ 2)
    _correlation_tensor!(C, ρ, matrix_basis)
    return C
end
export correlation_tensor

function _correlation_tensor!(tensor::Array{T, N}, ρ::AbstractMatrix{CT}, matrix_basis::NTuple{N, Vector{MB}}) where {T <: Real, CT <: Number, MB <: AbstractMatrix{Complex{T}}, N}
    dims2 = collect(length.(matrix_basis))
    vi = ones(Int, N)
    for i in 0:prod(dims2)-1
        tensor[vi...] = real(LA.dot(kron((matrix_basis[ni][di] for (ni, di) in enumerate(vi))...), ρ))
        _update_odometer!(vi, dims2)
    end
    return tensor
end

# copied from Ket, but changed the convention to start from 1 as we want indices
function _update_odometer!(ind::AbstractVector{<:Integer}, base::AbstractVector{<:Integer})
    ind[1] += 1
    d = length(ind)
    @inbounds for i in 1:d
        if ind[i] > base[i]
            ind[i] = 1
            i < d ? ind[i+1] += 1 : return
        else
            return
        end
    end
end

"""
    _correlation_tensor_ket!(tensor::Vector{T}, φ::Vector{Complex{T}}, matrix_basis)

Convert a ket `φ` to a correlation tensor Vector{T}, with same subspace dimension `d`.
"""
function _correlation_tensor_ket!(tensor::Vector{T}, φ::AbstractVector{Complex{T}}, matrix_basis::Vector{MB}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}}
    for i in eachindex(matrix_basis)
        tensor[i] = real(LA.dot(φ, LA.Hermitian(matrix_basis[i]), φ))
    end
    return tensor
end

"""
    function density_matrix(tensor::Vector{T}, dims::NTuple{N,Int}, matrix_basis = _gellmann(T, dims)) where {T <: Real} where {N}

Convert tensors of pure states to density matrix (for eigendecomposition).
"""
function density_matrix(tensor::AbstractArray{T, N}, dims::NTuple{N, Int}, matrix_basis = _gellmann(T, dims)) where {T <: Real, N}
    ρ = LA.Hermitian(Matrix{Complex{T}}(undef, prod(dims), prod(dims)))
    return _density_matrix!(ρ, tensor, matrix_basis)
end
export density_matrix

function density_matrix(ps::PureState{T, N}) where {T <: Real, N}
    dims = ps.lmo.dims
    ρ = LA.Hermitian(Matrix{Complex{T}}(undef, prod(dims), prod(dims)))
    return _density_matrix!(ρ, ps, ps.lmo.matrix_basis)
end

function _density_matrix!(ρ::Matrix{Complex{T}}, tensor::AbstractArray{T, 1}, matrix_basis::Vector{MB}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}}
    ρ .= 0
    for i in eachindex(matrix_basis), j in eachindex(matrix_basis[i])
        ρ[j] += tensor[i] * matrix_basis[i][j]
    end
    ρ ./= 2
    return ρ
end

function _density_matrix!(ρ::LA.Hermitian, tensor::AbstractArray{T, 1}, matrix_basis::NTuple{1, Vector{MB}}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}}
    _density_matrix!(parent(ρ), tensor, matrix_basis[1])
    return ρ
end

function _density_matrix!(ρ::LA.Hermitian, tensor::AbstractArray{T, N}, matrix_basis::NTuple{N, Vector{MB}}) where {T <: Real, MB <: AbstractMatrix{Complex{T}}, N}
    data = parent(ρ)
    data .= 0
    dims2 = collect(length.(matrix_basis))
    vi = ones(Int, N)
    for i in 0:prod(dims2)-1
        # kron allocates a lot, but this is fine for the moment (no performance critical function)
        data .+= tensor[vi...] * kron((matrix_basis[ni][di] for (ni, di) in enumerate(vi))...)
        _update_odometer!(vi, dims2)
    end
    data ./= 2^N
    return ρ # TODO confirm this is correct
end

"""
    _reduced_tensor!(tensor::Vector{T}, pure_tensors::NTuple{N, Vector{T}}, dir::Array{T, N}, j::Int, s::Vector{Int} = setdiff(1:N, j)) where {T <: Real, N}

Computes the correlation tensor of the `j`-th subsystem (the tensor-version of the partial trace).
When N=2, j=1, computes ⟨ϕ2|dir|ϕ2⟩ for dir ∈ H₁ ⊗ H₂
"""
function _reduced_tensor!(tensor::Vector{T}, pure_tensors::NTuple{N, Vector{T}}, dir::AbstractArray{T, N}, j::Int, s::Vector{Int} = setdiff(1:N, j)) where {T <: Real, N}
    tensor .= 0
    for ind in CartesianIndices(dir)
        b = one(T)
        for i in s
            b *= pure_tensors[i][ind[i]]
        end
        tensor[ind[j]] += b * dir[ind]
    end
end
