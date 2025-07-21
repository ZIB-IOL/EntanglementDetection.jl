@testset "Bell state (2 × 2)              " begin
    for R in [Float64, Double64]
        T = Complex{R}
        res = entanglement_detection(Ket.state_phiplus(T), (2, 2);
            fw_algorithm = FrankWolfe.blended_pairwise_conditional_gradient,
            epsilon = 1e-6,
            lazy = true,
            lazy_tolerance = 2.0,
            max_iteration = 10^5,
            verbose = false)
        @test res.ent
        @test real(LA.dot(res.witness.W, Ket.state_phiplus(T))) < 0
        @test real(LA.dot(res.witness.W, Ket.state_phiplus(T; v = 0))) > 0
        @test 0.0 ≤ real(LA.dot(res.witness.W, Ket.state_phiplus(T; v = 1 / 3))) < 0.0861
    end
end

@testset "Horodecki state (3 × 3)         " begin
    for R in [Float64, Double64]
        T = Complex{R}
        res = entanglement_detection(Ket.state_horodecki33(T, 0.27), (3, 3);
            fw_algorithm = FrankWolfe.blended_pairwise_conditional_gradient,
            epsilon = 1e-6,
            lazy = true,
            lazy_tolerance = 2.0,
            max_iteration = 10^4,
            verbose = 0)
        @test res.ent
        @test real(LA.dot(res.witness.W, Ket.state_horodecki33(T, 0.27))) < 0
        @test real(LA.dot(res.witness.W, Ket.state_horodecki33(T, 0.27; v = 0))) > 0
        #TODO maybe remove this test
        @test real(LA.dot(res.witness.W, Ket.state_horodecki33(T, 0.27; v = 0.988))) < 0 #v=0.937, upper bound for separability
    end
end
