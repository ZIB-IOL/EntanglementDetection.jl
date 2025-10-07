abstract type SeparableLMO{T, N} <: FrankWolfe.LinearMinimizationOracle end

"""
    Workspace{T, N}

Structure for initial pre-allocation of performance-critical functions.
"""
struct Workspace{T, N}
    pure_kets::NTuple{N, Vector{Complex{T}}} # vectors of pure product states
    pure_tensors::NTuple{N, Vector{T}} # tensors of pure product states
    reduced_setdiffs::NTuple{N, Vector{Int}} # used in _reduced_tensor!
    reduced_tensors::NTuple{N, Vector{T}} # tensors of reduced density matrices of one qudit
    reduced_matrices::NTuple{N, Matrix{Complex{T}}} # reduced density matrices of one qudit
    blas_workspaces::NTuple{N, BlasWorkspace{T}}
end

function Workspace{T, N}(dims::NTuple{N, Int}) where {T <: Real, N}
    pure_kets = ntuple(n -> Vector{Complex{T}}(undef, dims[n]), Val(N))
    pure_tensors = ntuple(n -> Vector{T}(undef, dims[n]^2), Val(N))
    reduced_setdiffs = ntuple(n -> setdiff(1:N, n), Val(N))
    reduced_tensors = ntuple(n -> Vector{T}(undef, dims[n]^2), Val(N))
    reduced_matrices = ntuple(n -> Matrix{Complex{T}}(undef, dims[n], dims[n]), Val(N))
    blas_workspaces = ntuple(n -> BlasWorkspace(T, dims[n]), Val(N))

    return Workspace{T, N}(pure_kets, pure_tensors, reduced_setdiffs, reduced_tensors, reduced_matrices, blas_workspaces)
end

struct Fwdata
    fw_iter::Vector{Int}
    fw_time::Vector{Float64}
    lmo_counts::Vector{Int}
end

function Fwdata()
    fw_iter = [1]
    fw_time = [0.0]
    lmo_counts = [0]
    return Fwdata(fw_iter, fw_time, lmo_counts)
end

"""
    AlternatingSeparableLMO{T, N, MB <: AbstractMatrix{Complex{T}}} <: SeparableLMO{T, N}

`AlternatingSeparableLMO` implements `compute_extreme_point(lmo, direction)` which returns a pure product state used in Frank-Wolfe algorithms.
The method used is an alternating algorithm starting from random pure states on each party and alternatively optimizing each reduced state via an eigendecomposition.

Type parameters:
- `T`: element type of the correlation tensor
- `N`: number of parties
- `MB`: type of the matrix basis

Fields:
- `dims`: dimensions of the reduced state on each party
- `matrix_basis`: matrix basis of the correlation tensor
- `max_iter`: maximum number of alternation steps
- `threshold`: threshold to stop the alternation
- `nb`: number of random rounds to find the possible global optimal
- `workspace`: contains fields pre-allocated performance-critical functions
- `tmp`: temporary vector for fast scalar products of bipartite tensors
"""
struct AlternatingSeparableLMO{T, N, MB <: AbstractMatrix{Complex{T}}} <: SeparableLMO{T, N}
    dims::NTuple{N, Int}
    matrix_basis::NTuple{N, Vector{MB}}
    max_iter::Int
    threshold::T
    nb::Int
    parallelism::Bool
    workspaces::Vector{Workspace{T, N}}
    fwdata::Fwdata
    tmp::Vector{T}
end

function AlternatingSeparableLMO(
    ::Type{T},
    dims::NTuple{N, Int};
    matrix_basis = _gellmann(Complex{T}, dims),
    max_iter = 10^2,
    threshold = Base.rtoldefault(T),
    nb = 20,
    parallelism = false,
    verbose = 0,
    kwargs...
) where {T <: Real, N}
    MB = typeof(matrix_basis[1][1])
    if verbose > 0
        println("Quantum state structure: ", N, "-partite with local dimensions ", dims)
        println("Device numerical accuracy is ", threshold)
        println("Data is stored as ", typeof(matrix_basis[1][1]))
        if parallelism
            println("Parallelism is enabled with ", Threads.nthreads(), " threads.")
        else
            println("Parallelism is disabled.")
        end
    end
    if parallelism
        prod(dims) < 16 && verbose > 0 && @warn "The system dimension is small, parallelism may not be effective."
        LA.BLAS.set_num_threads(1)
        workspaces = [Workspace{T, N}(dims) for _ in 1:Threads.nthreads()]
    else
        workspaces = [Workspace{T, N}(dims)]
    end
    fwdata = Fwdata()
    tmp = N == 2 ? Vector{T}(undef, dims[1]^2) : T[]
    return AlternatingSeparableLMO{T, N, MB}(dims, matrix_basis, max_iter, threshold, nb, parallelism, workspaces, fwdata, tmp)
end

"""
    EnumeratingSeparableLMO{T, N, MB <: AbstractMatrix{Complex{T}}} <: SeparableLMO{T, N}

`EnumeratingSeparableLMO` implements `compute_extreme_point(lmo, direction)` which returns a pure product state used in Frank-Wolfe algorithms.
The method used is the enumeration of a fine approximation of the sets of states on all parties except the first one, for which an eigendecomposition suffices.

Type parameters:
- `T`: element type of the correlation tensor
- `N`: number of parties
- `MB`: type of the matrix basis

Fields:
- `dims`: dimensions of the reduced state on each party
- `matrix_basis`: matrix basis of the correlation tensor
- `workspace`: contains fields pre-allocated performance-critical functions
- `tmp`: temporary vector for fast scalar products of bipartite tensors
"""
struct EnumeratingSeparableLMO{T, N, MB <: AbstractMatrix{Complex{T}}, Nminus1, A <: ApproximationQuantumStates{T}} <: SeparableLMO{T, N}
    dims::NTuple{N, Int}
    matrix_basis::NTuple{N, Vector{MB}}
    approximations::NTuple{Nminus1, A}
    workspace::Workspace{T, N}
    fwdata::Fwdata
    tmp::Vector{T}
end

function EnumeratingSeparableLMO(
    ::Type{T},
    dims::NTuple{N, Int};
    matrix_basis = _gellmann(T, dims),
    max_length = 10^6,
    min_η = zero(T),
    kwargs...
) where {T <: Real, N}
    max_length_per_party = max_length^(1 / (N - 1))
    approximations = (CrossPolytopeSubdivision{T}(dims[n]; max_length = max_length_per_party) for n in 2:N)
    η = prod([collect(approximations)[i].η for i in 1:N-1])
    if η < min_η
        @warn "The max_length = $max_length can not guarantee the minimum η = $min_η. Current η is $η."
    end
    return EnumeratingSeparableLMO(T, dims, approximations...)
end

function EnumeratingSeparableLMO(
    ::Type{T},
    dims::NTuple{N, Int},
    approximations::Vararg{A};
    matrix_basis = _gellmann(T, dims),
    kwargs...
) where {T <: Real, A <: ApproximationQuantumStates{T}, N}
    @assert length(approximations) == N - 1
    for n in 2:N
        @assert dims[n] == approximations[n-1].d
    end
    MB = typeof(matrix_basis[1][1])
    workspace = Workspace{T, N}(dims)
    fwdata = Fwdata()
    tmp = N == 2 ? Vector{T}(undef, dims[1]^2) : T[]
    return EnumeratingSeparableLMO{T, N, MB, N-1, A}(dims, matrix_basis, approximations, workspace, fwdata, tmp)
end

function radius_inner(lmo::EnumeratingSeparableLMO{T, N, MB, Nminus1, A}) where {T, N, MB, Nminus1, A}
    return prod(lmo.approximations[n].η for n in 1:Nminus1)
end

function radius_inner(lmo::AlternatingSeparableLMO{T}) where {T <: Real}
    return one(T)
end

"""
    KSeparableLMO{T, N, LMO} <: SeparableLMO{T, N}

KSeparableLMO implements `compute_extreme_point(lmo::LMO, direction)` which returns a pure product state by considering all K-partite partitions by iterating over all LMO{T, K}.

Type parameters:
- `T`: element type of the correlation tensor
- `N`: number of parties
- `MB`: type of the matrix basis
- `LMO`: type of the LMO for k-partite separability

Fields:
- `dims`: dimensions of the reduced state on each party
- `matrix_basis`: matrix basis of the correlation tensor
- `lmos`: list of LMOs for all K-partitions
- `partitions`: list of K-partite partitions
"""
struct KSeparableLMO{T, N, MB <: AbstractMatrix{Complex{T}}, LMO <: SeparableLMO{T}} <: SeparableLMO{T, N}
    dims::NTuple{N, Int} # need for initialization in separable_distance
    matrix_basis::NTuple{N, Vector{MB}}
    lmos::Vector{LMO}
    partitions::Vector{Vector{Vector{Int}}}
    fwdata::Fwdata
end

function KSeparableLMO(
    ::Type{T},
    dims::NTuple{N, Int},
    k::Int = 2;
    matrix_basis = _gellmann(T, dims),
    LMO = AlternatingSeparableLMO,
    kwargs...
) where {T <: Real, N}
    @assert k ≤ N "The number of parties should be larger than the k-separability problem."
    N == k && return AlternatingSeparableLMO(T, dims; matrix_basis, kwargs...)
    MB = typeof(matrix_basis[1][1])
    # GME only need consider all bipartitions (k=2)
    partitions = _partitions(1:N, k)
    new_dims = Vector{Vector{Int}}(undef, length(partitions))
    lmos = Vector{LMO{T, k, MB}}(undef, length(partitions))
    for i in eachindex(partitions)
        new_dims[i] = [prod(dims[partitions[i][j]]) for j in eachindex(partitions[i])]
        lmos[i] = LMO(T, Tuple(new_dims[i]); matrix_basis = reconstruct_basis(matrix_basis, partitions[i]), kwargs...)
    end
    fwdata = Fwdata()
    return KSeparableLMO{T, N, MB, LMO{T, k, MB}}(dims, matrix_basis, lmos, partitions, fwdata)
end

"""
    PureState{T, N} <: AbstractArray{T, N}

Represents a pure product state. Each subsystem is a pure state stored as a tensor PureState.tensors[n].
"""
struct PureState{T, N, LMO} <: AbstractArray{T, N}
    tensors::NTuple{N, Vector{T}} # correlation tensors of the individual parties
    obj::T # =<`tensors`,∇>, the minimal real eigenvalue of gradient direction
    lmo::LMO # gives access to tmp (for bipartite scalar product) and the matrix basis used
end

function PureState(x::PureState{T, N, LMO}) where {T, N, LMO}
    return PureState{T, N, LMO}(x.tensors, x.obj, x.lmo)
end

Base.IndexStyle(::Type{<:PureState}) = IndexCartesian()
Base.size(ps::PureState) = Tuple(length.(ps.tensors))

Base.@propagate_inbounds function Base.getindex(ps::PureState{T, 2}, x::Vararg{Int, 2}) where {T <: Real}
    @boundscheck (checkbounds(ps, x...))
    return @inbounds getindex(ps.tensors[1], x[1]) * getindex(ps.tensors[2], x[2])
end

Base.@propagate_inbounds function Base.getindex(ps::PureState{T, N}, x::Vararg{Int, N}) where {T <: Real, N}
    @boundscheck (checkbounds(ps, x...))
    return @inbounds prod(getindex(ps.tensors[n], x[n]) for n in 1:N)
end

LA.dot(A::Array, ps::PureState) = conj(LA.dot(ps, A))

function LA.dot(ps::PureState{T, 2}, A::Array{T, 2}) where {T <: Real}
    LA.mul!(ps.lmo.tmp, A, ps.tensors[2])
    return LA.dot(ps.tensors[1], ps.lmo.tmp)
end

function LA.dot(ps1::PureState{T, 2}, ps2::PureState{T, 2}) where {T <: Real}
    return LA.dot(ps1.tensors[1], ps2.tensors[1]) * LA.dot(ps1.tensors[2], ps2.tensors[2])
end

function LA.dot(ps1::PureState{T, N}, ps2::PureState{T, N}) where {T <: Real, N}
    return prod(LA.dot(ps1.tensors[n], ps2.tensors[n]) for n in 1:N)
end
