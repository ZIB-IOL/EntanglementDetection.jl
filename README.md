# EntanglementDetection

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zib-iol.github.io/EntanglementDetection.jl/dev/)
[![Build Status](https://github.com/ZIB-IOL/EntanglementDetection.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ZIB-IOL/EntanglementDetection.jl/actions/workflows/ci.yml)

This package addresses the **entanglement** and **separability** certification for multipartite quantum systems with arbitrary local dimensions.

The original article for which it was written can be found here:

> [1] [A Unified Toolbox for Multipartite Entanglement Certification](https://arxiv.org/abs/2507.17435).

The method for separability certification as the part of the package was first introduced in

> [2] [Convex optimization over classes of multiparticle entanglement](https://arxiv.org/abs/1707.02958).

## Installation

The most recent release is available via the julia package manager, e.g., with

```julia
using Pkg
Pkg.add("EntanglementDetection")
```

or the main branch:

```julia
Pkg.add(url="https://github.com/ZIB-IOL/EntanglementDetection.jl", rev="main")
```

## Getting started

Let's say we want to analyze the entanglement property of the two-qubit maximally entangled state with white noise.
Using `EntanglementDetection.jl`, here is what the code looks like.

```julia
julia> using EntanglementDetection, LinearAlgebra, Ket

julia> d = 2; # qubit system

julia> N = 2; # bipartite scenario

julia> p = 0.2; # white noise strength

julia> ρ = Ket.state_ghz(d, N; v = 1 - p) # two-qubit maximally entangled state with white noise
4×4 Hermitian{ComplexF64, Matrix{ComplexF64}}:
 0.45+0.0im   0.0+0.0im   0.0+0.0im   0.4+0.0im
  0.0-0.0im  0.05+0.0im   0.0+0.0im   0.0+0.0im
  0.0-0.0im   0.0-0.0im  0.05+0.0im   0.0+0.0im
  0.4-0.0im   0.0-0.0im   0.0-0.0im  0.45+0.0im

julia> dims = Tuple(fill(d, N)); # the entanglement structure 2 × 2

julia> res = separable_distance(ρ, dims); # achieve the distance to the separable space 
Iteration        Primal    Dual gap     #Atoms
        1    1.6600e+00    4.0000e+00         1
    10000    3.2667e-01    6.1346e-08         10
    20000    3.2667e-01    6.1346e-08         10
    30000    3.2667e-01    6.1346e-08         10
    40000    3.2667e-01    6.1346e-08         10
    50000    3.2667e-01    6.1346e-08         10
    60000    3.2667e-01    6.1346e-08         10
    70000    3.2667e-01    6.1346e-08         10
    80000    3.2667e-01    6.1346e-08         10
    90000    3.2667e-01    6.1346e-08         10
   100000    3.2667e-01    6.1346e-08         10
     Last    3.2667e-01    9.3576e-08         10
[ Info: Stop: maximum iteration reached
```

For the state ``ρ``, as the distance to the separable space `res.primal` is much larger than 0, practically, we can detect the entanglement of the state with confidence (technically speaking, ``Primal`` $\gg$ ``Dual gap``.)

## Entanglement certification

In principle, if ``Primal`` $\geq$ ``Dual gap``, the state is outside the separable space, therefore is entangled. However, due to the heuristic method, the ``Dual gap`` is inaccuracy. In practice, we can detect the entanglement by check enlarging the factor, e.g., ``Primal`` $\geq 5 \times$ ``Dual gap``.

A rigorous tool is also introduce in our package:

```julia
julia> witness = entanglement_witness(ρ, res.σ, dims); # construct a rigorous entanglement witness

julia> real(dot(witness.W, ρ)) < 0 # if Tr(Wρ) < 0, then the state ρ is entangled
true
```

## Separability certification

Let's consider the other case that there is more noise mixed in the state.

```julia
julia> d = 2; N = 2; p = 0.8; ρ = Ket.state_ghz(d, N; v = 1 - p) # with more white noise
4×4 Hermitian{ComplexF64, Matrix{ComplexF64}}:
 0.3+0.0im  0.0+0.0im  0.0+0.0im  0.1+0.0im
 0.0-0.0im  0.2+0.0im  0.0+0.0im  0.0+0.0im
 0.0-0.0im  0.0-0.0im  0.2+0.0im  0.0+0.0im
 0.1-0.0im  0.0-0.0im  0.0-0.0im  0.3+0.0im

julia> res = separable_distance(ρ, dims); # achieve the distance to the separable space 
   Iteration        Primal    Dual gap     #Atoms
           1    1.3600e+00    4.0000e+00          1
        Last    7.7981e-07    1.0612e-03         14
[ Info: Stop: primal small enough
```

For this case, ``Primal`` is much smaller than ``Dual gap``, which can not be detected as an entangled state, and also can not be confirmed by entanglement witness:

```julia
julia> witness = entanglement_witness(ρ, res.σ, dims); # construct a rigorous entanglement witness

julia> real(dot(witness.W, ρ)) < 0 # if Tr(Wρ) > 0, then the state ρ could be entangled or separable.
false
```

In order to certify separability, a geometric reconstruction procedure was introduced in Ref. [2] as the part of our package to build a unified toolbox for entanglement analysis:

```julia
julia> sep = separability_certification(ρ, dims; verbose = 0); # certify separability by geometric reconstruction

julia> sep.sep
true
```

## Under the hood

The computation is based on an efficient variant of the Frank-Wolfe algorithm to iteratively find the separable state closest to the input quantum state based on correlation tensor.
See this recent [review](https://arxiv.org/abs/2211.14103) for an introduction to the method and the package [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl) for the implementation on which this package relies.

## Going further

More examples can be found in the corresponding folder of the package.
They include the application on the 10-qubit system with shortcut method and multipartite systems with different entanglement structures.
