# using shortcut mode, we can speed up the computation thus to handle larger systems
using EntanglementDetection
using FrankWolfe
using Ket
import LinearAlgebra as LA
using HiGHS
import MathOptInterface as MOI
using Random
using Serialization

output_dir = "EntanglementDetection/examples/results/10qubits"


max_length = 10^8
T = Float64

# Initial the entanglement structure
N = 10
dims = Tuple(fill(2, N))
lmo = EntanglementDetection.AlternatingSeparableLMO(T, dims; nb = 10, threshold = Base.rtoldefault(T), max_iter = 10^3, parallelism = true)

Random.seed!(0)
println("Initializing the state ...")
v = 0.3
ρ = Ket.state_ghz(Complex{T}, N; v)
println("Converting ...")
C = correlation_tensor(ρ, dims)

println("Beginning the computation ...")
# Using the Frank-Wolfe algorithm with LP solver
ini_tensor = similar(C)
ini_tensor .= 0
ini_tensor[1] = one(T)
v0 = FrankWolfe.compute_extreme_point(lmo, ini_tensor - C)
o = MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true))
active_set = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSetQuadraticProductCaching([(one(T), copy(v0))], LA.I, -C),
    LA.I,
    -C,
    o,
    wolfe_step=true
    )

res = separable_distance(C, lmo;
    fw_algorithm = FrankWolfe.blended_pairwise_conditional_gradient,
    verbose = 2,
    active_set = active_set,
    max_iteration = 10^3,
    callback_iter = 2,
    recompute_last_vertex = false,
    epsilon = 1e-8,
    shortcut = true,
    shortcut_scale = 10,
    trajectory = true,
)

abs_output =  string(output_dir , "/ghz_" ,N, ".dat")
serialize(abs_output, res.x)

traj = res.traj_data
n_points = 10^3
indices = round.(Int, range(1, length(traj)-1, length=n_points))
itr = [traj[i][1] for i in indices]
primal = [traj[i][2] for i in indices]
dual_gap = [traj[i][4] for i in indices]
time = [traj[i][5] for i in indices]

plot_data = (itr, primal, dual_gap, time)
plot_dir =  string(output_dir , "/ghz_" ,N , "_plot.dat")
serialize(plot_dir, plot_data)
