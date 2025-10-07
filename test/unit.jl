@testset "Conversions                     " begin
    for T in [Float64, Double64, Float128, BigFloat]
        CT = Complex{T}
        for rank in [1, 2], dims in [(2, 2), (2, 3), (3, 3), (2, 2, 2)]
            N = length(dims)
            ρ = Matrix(Ket.random_state(CT, prod(dims), rank == 1 ? 1 : prod(dims)))
            tensor = EntanglementDetection.correlation_tensor(ρ, dims)
            @test tensor[1]^2 ≈ 2^N / prod(dims)
            if rank == 1
                @test LA.norm(tensor)^2 ≈ 2^N
            else
                @test LA.norm(tensor)^2 ≤ 2^N
            end
            ρ2 = EntanglementDetection.density_matrix(tensor, dims)
            @test ρ2 ≈ ρ
        end
    end
end

@testset "LMOSeesaw                       " begin
    for T in [Float64, Double64, Float128, BigFloat]
        CT = Complex{T}
        for dims in [(2, 2), (2, 3), (3, 3), (2, 2, 2)]
            N = length(dims)
            lmo = EntanglementDetection.AlternatingSeparableLMO(T, dims)
            @test isa(lmo, EntanglementDetection.SeparableLMO{T, N})
            @test isa(lmo, EntanglementDetection.AlternatingSeparableLMO{T, N})
            @test lmo.dims == dims
            @test lmo.matrix_basis[1][2:end] == Ket.gellmann.(CT, dims)[1][2:end]
            @test lmo.matrix_basis[2][1] == Ket.gellmann.(CT, dims[2])[1] * sqrt(T(2)) / sqrt(T(dims[2]))
            ρ = Matrix(Ket.proj(CT, 1, prod(dims)))
            v = FrankWolfe.compute_extreme_point(lmo, EntanglementDetection.correlation_tensor(LA.I / prod(dims) - ρ, dims))
            @test EntanglementDetection.density_matrix(v) ≈ ρ
        end
    end
end

@testset "Basis transformations           " begin
    for T ∈ [Float64, Double64, Float128, BigFloat]
        CT = Complex{T}
        dims = (2, 2, 2)
        N = length(dims)
        kets = Ket.random_state_ket.(CT, dims)
        ρ = kron(kets...) * kron(kets...)'
        lmo = EntanglementDetection.AlternatingSeparableLMO(T, dims)
        tensor = EntanglementDetection.correlation_tensor(ρ, dims)
        tensors = ntuple(n -> Vector{T}(undef, dims[n]^2), Val(N))
        for i in 1:N
            EntanglementDetection._correlation_tensor_ket!(tensors[i], kets[i], lmo.matrix_basis[i])
        end
        ps = EntanglementDetection.PureState{T, N, typeof(lmo)}(tensors, 0, lmo)
        for i in CartesianIndices(tensor)
            @test tensor[i] ≈ ps.tensors[1][i[1]] * ps.tensors[2][i[2]] * ps.tensors[3][i[3]]
        end
        @test EntanglementDetection.ungroup_dims(ps, dims, [[1, 2, 3]]) ≈ tensor
        partition = [[1; collect(3:length(dims))], [2]]
        dims2 = (prod(Int, dims[partition[1]]), prod(Int, dims[partition[2]]))
        lmo2 = EntanglementDetection.AlternatingSeparableLMO(T, dims2; matrix_basis = EntanglementDetection.reconstruct_basis(lmo.matrix_basis, partition))
        tensor2 = EntanglementDetection.group_dims(tensor, dims2, partition)
        ps2 = EntanglementDetection.PureState{T, 2, typeof(lmo2)}((kron(tensors[3],tensors[1]), tensors[2]), 0, lmo2)
        @test EntanglementDetection.ungroup_dims(ps2, dims, partition) ≈ tensor
        for i in CartesianIndices(tensor)
            @test tensor[i] == tensor2[i[1] - 1 + (i[3] - 1) * 2^2 + 1, i[2]]
            @test kron(lmo.matrix_basis[1][i[1]],lmo.matrix_basis[3][i[3]]) == lmo2.matrix_basis[1][i[1] - 1 + (i[3] - 1) * 2^2 + 1]
        end
        @test tensor == EntanglementDetection.ungroup_dims(tensor2, dims, partition)
        ρ2 = EntanglementDetection.density_matrix(EntanglementDetection.ungroup_dims(tensor2, dims, partition), dims, lmo.matrix_basis)
        @test ρ2 ≈ ρ
    end
end

@testset "KSeparableLMO                   " begin
    for T in [Float64, Double64, Float128, BigFloat]
        CT = Complex{T}
        for dims in [(2, 2, 2), (2, 2, 3), (2, 3, 3)]
            N = length(dims)
            lmo = EntanglementDetection.KSeparableLMO(T, dims, 2)
            @test isa(lmo, EntanglementDetection.SeparableLMO{T, N})
            @test isa(lmo, EntanglementDetection.KSeparableLMO{T, N})
            @test length(lmo.partitions) == 3
            @test [[1, 2], [3]] in lmo.partitions
            @test [[1, 3], [2]] in lmo.partitions
            @test [[1], [2, 3]] in lmo.partitions
        end
    end
end

@testset "Separable distance              " begin
    for T in [Float64, Double64, Float128, BigFloat]
        CT = Complex{T}
        for dims in [(2, 2), (2, 3), (3, 3), (2, 2, 2)]
            N = length(dims)
            ρ = Matrix{CT}(LA.I, prod(dims), prod(dims))
            ρ ./= prod(dims)
            σ, v, primal, noise_level, active_set, lmo = separable_distance(ρ, dims; verbose = 0)
            @test isa(σ, AbstractMatrix{CT})
            @test isa(v, EntanglementDetection.PureState{T, N})
            @test isa(primal, T)
            @test isa(lmo, EntanglementDetection.SeparableLMO{T, N})
            @test abs(primal - LA.norm(ρ - σ)^2 / 2) < Base.rtoldefault(T)
            @test primal < 1e-6 / 2^N
        end
    end
end

@testset "Different algorithms            " begin
    algorithms = [
                  FrankWolfe.blended_pairwise_conditional_gradient,
                  FrankWolfe.frank_wolfe,
                  FrankWolfe.away_frank_wolfe,
                  # FrankWolfe.lazified_conditional_gradient, # FrankWolfe.VectorCacheLMO ruins the game
                 ]
    for T in [Float64, Double64]
        CT = Complex{T}
        for dims in [(2, 2), (2, 3), (2, 2, 2)], fw_algorithm in algorithms
            ρ = Matrix{CT}(LA.I, prod(dims), prod(dims))
            ρ ./= prod(dims)
            res = separable_distance(ρ, dims; fw_algorithm, verbose = 0)
            @test res.primal < 1e-6
            @test res.σ ≈ ρ
        end
    end
end
