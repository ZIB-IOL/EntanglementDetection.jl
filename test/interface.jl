using EntanglementDetection
import Ket
import LinearAlgebra as LA

# Problem setting
p = 0.888 # 1 - 1/9
T = Float64
dims = [2, 2, 2, 2]
target_rho = LA.Hermitian((1 - p) * Ket.state_ghz(Complex{T}, 4) + p * Complex.(Matrix{T}(LA.I, prod(dims), prod(dims)) / prod(dims)))

# Some oracle give us a guess

guess_rho = Complex.(Matrix{T}(LA.I, prod(dims), prod(dims)) / prod(dims))

# Continue with the guess
max_iteration = 1e5
epsilon = 1e-6
res = separable_distance(target_rho; dims, ini_sigma = guess_rho, max_iteration, epsilon)
res.primal
HS_norm = sqrt(res.primal)
