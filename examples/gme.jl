# using lmo to define the entanglement structure
# we supply the ``EntanglementDetection.KSeparableLMO`` for K-separability structure, ``k=2'' corresponding to GME
# this is example is to achieve the entanglement and separability bound for robustness problem under GME
using EntanglementDetection
using FrankWolfe
using Ket
using LinearAlgebra
using HiGHS
import MathOptInterface as MOI
using Serialization

max_iteration = 10^7    # control time
callback_iter = 10^3
max_active = 4 * 10^4   # control the size of memory
shortcut_scale = 10
noise_atol = 1e-3

output_dir = "EntanglementDetection/examples/results/gme"

T = Float64
N = 4
dims = Tuple(fill(2, N))
ρ = Ket.state_ghz(Complex{T}, N)

k = 2
C = correlation_tensor(ρ, dims)
lmo = EntanglementDetection.KSeparableLMO(T, dims, k; nb = 10, threshold = Base.rtoldefault(T), max_iter = 10^3, parallelism = true)

# the noise_mixture mode can fast reach the boundary of the separable set (outside)
res = separable_distance(C, lmo;
    fw_algorithm = FrankWolfe.blended_pairwise_conditional_gradient,
    verbose = 2,
    max_iteration = max_iteration,
    callback_iter = callback_iter,
    recompute_last_vertex = false,
    epsilon = 1e-7,
    noise_mixture = true,
    noise_atol = noise_atol,
    shortcut = true,
    shortcut_scale = shortcut_scale,
    trajectory = false,
    max_active = max_active,
)

ρ_noise(p) = (1-p) * ρ + p * Matrix{Complex{T}}(I, prod(dims), prod(dims)) / prod(dims)

Δp = zero(T)
sep = (sep = false, res = (active_set = nothing, primal = zero(T)), r = T[], ϵ = T[])
for p in res.noise_level+noise_atol:noise_atol:1.0
    println("Testing p = ", p)
    sep = separability_certification(ρ_noise(p), dims, lmo; 
        certify_test = 30,
        max_iteration = max_iteration,
        epsilon = 1e-7,
        shortcut = true,
        trajectory = false,
        max_active = max_active,
        )
    if sep.sep
        Δp = p - res.noise_level
        break
    end
end
println(Δp)

txt_output =  string(output_dir, "/results_", k, ".txt")
open(txt_output, "w") do io
    p = res.noise_level
    write(io, "$p\n$Δp")
end
