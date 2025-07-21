abstract type ApproximationQuantumStates{T} end

"""
    PolytopeReducedComplexSphere{T} <: ApproximationQuantumStates{T}

Approximates the complex sphere in dimension `d` via a polytope on the real sphere in dimension `2d-1`.
"""
mutable struct PolytopeReducedComplexSphere{T} <: ApproximationQuantumStates{T}
    η::T
    d::Int
    prev::Int
    mat::Matrix{T} # rows contain the vertices of the polytope on the real sphere in dimension 2d-1
end
export PolytopeReducedComplexSphere

function PolytopeReducedComplexSphere{T}(mat::Matrix, shr) where {T <: Real}
    @assert isodd(length(size(mat, 2)))
    d = (size(mat, 2) + 1) ÷ 2
    return PolytopeReducedComplexSphere{T}(2shr^2 - 1, d, 0, mat)
end

Base.length(A::PolytopeReducedComplexSphere) = size(A.mat, 1)

function _populate!(ket::Vector{Complex{T}}, A::PolytopeReducedComplexSphere{T}, i::Int) where {T <: Real}
    i == A.prev && return ket
    ket[1] = A.mat[i, 1]
    for j in 2:A.d
        ket[j] = Complex(A.mat[i, j], A.mat[i, j+A.d-1])
    end
    A.prev = i
    return ket
end

"""
    PolytopePhasedComplexSphere{T} <: ApproximationQuantumStates{T}

Approximates the complex sphere in dimension `d` via a polytope on the real sphere in dimension `d` and `d-1` phases.
"""
mutable struct PolytopePhasedComplexSphere{T} <: ApproximationQuantumStates{T}
    η::T
    d::Int
    prev::Int
    mat::Matrix{T} # rows contain the vertices of the polytope on the real sphere in dimension d
    n::Int # sets the number of the angular subdivision of the d-1 phases (angle π/n)
    # pre-allocated fields
    ind::Vector{Int}
    base::Vector{Int}
    sincos::Vector{Tuple{T, T}}
end
export PolytopePhasedComplexSphere

function PolytopePhasedComplexSphere{T}(mat::Matrix, shr, n) where {T <: Real}
    d = size(mat, 2)
    shr *= cos(T(pi) / T(2n))^(d - 1)
    ind = ones(Int, d)
    base = [size(mat, 1); fill(n, d - 1)]
    sc = sincos.((T(pi) / n) * (0:n-1))
    return PolytopePhasedComplexSphere{T}(2shr^2 - 1, d, 0, mat, n, ind, base, sc)
end

Base.length(A::PolytopePhasedComplexSphere) = size(A.mat, 1) * A.n^(A.d - 1)

function _populate!(ket::Vector{Complex{T}}, A::PolytopePhasedComplexSphere{T}, i::Int) where {T <: Real}
    i == A.prev && return ket
    ket .= A.mat[A.ind[1], :]
    for j in 2:A.d
        ket[j] *= Complex(A.sincos[A.ind[j]][2], A.sincos[A.ind[j]][1])
    end
    _update_odometer!(A.ind, A.base)
    A.prev = i
    return ket
end

"""
    CrossPolytopeSubdivision{T} <: ApproximationQuantumStates{T}

Approximates the complex sphere in dimension `d` via an edgewise subdivision of the cross-polytope on the real sphere in dimension `2d-1`.
"""
mutable struct CrossPolytopeSubdivision{T} <: ApproximationQuantumStates{T}
    η::T
    d::Int
    prev::Int
    n::Int # sets the number of the edgewise subdivision of the cross-polytope
    vec::Vector{Int}
    sgn::Vector{Int8}
end
export CrossPolytopeSubdivision

function CrossPolytopeSubdivision{T}(d::Integer; max_length = 10^6) where {T <: Real}
    D = 2d - 2
    n = round(Int, (factorial(D) * max_length)^(1 / D) / 2) - D + 1
    while binomial(n + D, D) * 2^D ≤ max_length
        n += 1
    end
    n -= 1
    return CrossPolytopeSubdivision{T}(d, n)
end

function CrossPolytopeSubdivision{T}(d::Integer, n::Integer) where {T <: Real}
    η = n / sqrt(T(n^2 + (d ≤ 3 ? 1 : 2) * (d - 1)))
    vec = fill(1, 2d - 1)
    vec[2d-1] = n + 1
    sgn = fill(Int8(-1), 2d - 2)
    return CrossPolytopeSubdivision{T}(η, d, 0, n, vec, sgn)
end

# this is the actual number, for which I'm missing an explicit mapping enumerating the vertices
# Base.length(A::CrossPolytopeSubdivision) = sum(binomial(2A.d - 1, i) * binomial(2A.d - 1 + A.n - i - 1, A.n - i) for i in 0:2A.d-1) ÷ 2
# this is a decent upper bound, with redundancy on the edges only
Base.length(A::CrossPolytopeSubdivision) = binomial(A.n + 2A.d - 2, 2A.d - 2) * 2^(2A.d - 2)

function _populate!(ket::Vector{Complex{T}}, A::CrossPolytopeSubdivision{T}, i::Int) where {T <: Real}
    i == A.prev && return ket
    if _next_composition!(A.vec)
        _next_signs!(A.sgn)
    end
    ket[1] = A.vec[1] - 1
    @inbounds for j in 2:A.d
        ket[j] = Complex(A.sgn[j-1] * (A.vec[j] - 1), A.sgn[j+A.d-2] * (A.vec[j+A.d-1] - 1))
    end
    ket ./= A.n
    LA.normalize!(ket)
    A.prev = i
    return ket
end

function _next_signs!(sgn::AbstractVector{<:Integer})
    sgn[1] += 2
    d = length(sgn)
    @inbounds for i in 1:d
        if sgn[i] > 1
            sgn[i] = -1
            i < d ? sgn[i+1] += 2 : return
        else
            return
        end
    end
end

# mostly copied from Combinat (c838e67)
function _next_composition!(u::AbstractVector{<:Integer})
    k = length(u)
    @inbounds for i in k:-1:2
        s = i - k - 1
        for j in i:length(u)
            s += u[j]
        end
        if s ≥ 1
            u[i-1] += 1
            u[k] = s
            u[i:k-1] .= 1
            return false
        end
    end
    # cycle when finished
    u[k] = u[1]
    u[1] = 1
    return true
end
