const syev_switch = 5

mutable struct BlasWorkspace{T}
    d::Int
    m::Base.RefValue{LA.BlasInt} # only for syevr
    W::Vector{T}
    Z::Matrix{Complex{T}}        # only for syevr
    isuppz::Vector{LA.BlasInt}   # only for syevr
    work::Vector{Complex{T}}
    lwork::LA.BlasInt
    rwork::Vector{T}
    lrwork::LA.BlasInt           # only for syevr
    iwork::Vector{LA.BlasInt}    # only for syevr
    liwork::LA.BlasInt           # only for syevr
    info::Base.RefValue{LA.BlasInt}
end

function BlasWorkspace(::Type{Float64}, d::Int)
    if d ≤ syev_switch
        W = Vector{Float64}(undef, d)
        work = Vector{ComplexF64}(undef, 33d)
        lwork = LA.BlasInt(33d)
        rwork = Vector{Float64}(undef, 3d-2)
        info = Ref{LA.BlasInt}()
        # dummy values for syevr specific fields
        m = Ref{LA.BlasInt}()
        Z = Matrix{ComplexF64}(undef, 0, 0)
        isuppz = Vector{LA.BlasInt}(undef, 0)
        lrwork = LA.BlasInt(0)
        iwork = Vector{LA.BlasInt}(undef, 0)
        liwork = LA.BlasInt(0)
    else
        m = Ref{LA.BlasInt}()
        W = Vector{Float64}(undef, d)
        Z = Matrix{ComplexF64}(undef, d, d)
        isuppz = Vector{LA.BlasInt}(undef, 2d)
        work = Vector{ComplexF64}(undef, 33d)
        lwork = LA.BlasInt(33d)
        rwork = Vector{Float64}(undef, 24d)
        lrwork = LA.BlasInt(24d)
        iwork = Vector{LA.BlasInt}(undef, 10d)
        liwork = LA.BlasInt(10d)
        info = Ref{LA.BlasInt}()
    end
    return BlasWorkspace{Float64}(d, m, W, Z, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)
end

function BlasWorkspace(::Type{T}, d::Int) where {T <: Real}
    m = Ref{LA.BlasInt}()
    W = Vector{T}(undef, 0)
    Z = Matrix{Complex{T}}(undef, 0, 0)
    isuppz = Vector{LA.BlasInt}(undef, 0)
    work = Vector{Complex{T}}(undef, 0)
    lwork = LA.BlasInt(0)
    rwork = Vector{T}(undef, 0)
    lrwork = LA.BlasInt(0)
    iwork = Vector{LA.BlasInt}(undef, 0)
    liwork = LA.BlasInt(0)
    info = Ref{LA.BlasInt}()
    return BlasWorkspace{T}(d, m, W, Z, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)
end

function _syev!(A::Matrix{ComplexF64}, ws::BlasWorkspace{Float64})
    ccall((LA.BLAS.@blasfunc(zheev_), Base.liblapack_name), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{LA.BlasInt}, Ptr{ComplexF64},
           Ref{LA.BlasInt}, Ptr{Float64}, Ptr{ComplexF64}, Ref{LA.BlasInt},
           Ptr{Float64}, Ptr{LA.BlasInt}, Clong, Clong),
          'V', 'U', ws.d, A, stride(A, 2), ws.W, ws.work, ws.lwork, ws.rwork, ws.info, 1, 1)
    return ws.W, A
end

function _syevr!(A::AbstractMatrix{ComplexF64}, ws::BlasWorkspace{Float64})
    ccall((LA.BLAS.@blasfunc(zheevr_), Base.liblapack_name), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{LA.BlasInt}, Ptr{ComplexF64},
           Ref{LA.BlasInt}, Ref{ComplexF64}, Ref{ComplexF64}, Ref{LA.BlasInt},
           Ref{LA.BlasInt}, Ref{ComplexF64}, Ptr{LA.BlasInt}, Ptr{Float64},
           Ptr{ComplexF64}, Ref{LA.BlasInt}, Ptr{LA.BlasInt}, Ptr{ComplexF64},
           Ref{LA.BlasInt}, Ptr{Float64}, Ref{LA.BlasInt}, Ptr{LA.BlasInt},
           Ref{LA.BlasInt}, Ptr{LA.BlasInt}, Clong, Clong, Clong),
          'V', 'I', 'U', ws.d, A, stride(A, 2), 0.0, 0.0, 1, 1, -1.0, ws.m, ws.W, ws.Z, ws.d,
          ws.isuppz, ws.work, ws.lwork, ws.rwork, ws.lrwork, ws.iwork, ws.liwork, ws.info, 1, 1, 1)
    return ws.W, ws.Z
end

"""
    _eigmin!(ket::Vector, matrix::Matrix)

Computes the minimal real eigenvalue and updates `ket` in place
The variable `matrix` of size d × d also gets overwritten.
For BLAS-compatible types, uses `LAPACK.syev!` for d ≤ 5 and `LAPACK.syevr!` otherwise.
For other types, falls back to `eigen!`.
"""
function _eigmin!(ket::Vector{ComplexF64}, matrix::Matrix{ComplexF64}, ws::BlasWorkspace{Float64})
    λ, X = ws.d ≤ syev_switch ? _syev!(matrix, ws) : _syevr!(matrix, ws)
    ket .= view(X, :, 1)
    return λ[1]
end

# https://www.netlib.org/lapack/lawnspdf/lawn183.pdf
# function _eigmin!(ket::Vector{Complex{T}}, matrix::Matrix{Complex{T}}, ws::BlasWorkspace{T}) where {T <: LA.BlasFloat}
    # full decomposition EVR, fallback of eigen! (slowest)
    # λ, X = LA.LAPACK.syevr!('V', 'A', 'U', matrix, 0.0, 0.0, 0, 0, -1.0)
    # minimal eigenvalue EVR (fastest for d ≥ 6)
    # λ, X = LA.LAPACK.syevr!('V', 'I', 'U', matrix, 0.0, 0.0, 1, 1, -1.0)
    # full decomposition EVD (better than EVR for d ≤ 4)
    # λ, X = LA.LAPACK.syevd!('V', 'U', matrix)
    # full decomposition EV (fastest for d ≤ 5)
    # λ, X = LA.LAPACK.syev!('V', 'U', matrix)
    # ket .= view(X, :, 1)
    # return λ[1]
# end

function _eigmin!(ket::Vector{Complex{T}}, matrix::Matrix{Complex{T}}, ws::BlasWorkspace{T}) where {T <: Real}
    F = LA.eigen!(LA.Hermitian(matrix))::LA.Eigen{Complex{T}, T, Matrix{Complex{T}}, Vector{T}}
    ket .= view(F.vectors, :, 1)
    return real(F.values[1])
end

