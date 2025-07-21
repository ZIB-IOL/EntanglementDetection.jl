function separability_certification(C::Array{T, N},
    lmo::SeparableLMO{T, N}; active_set=nothing, certify_test = 100, ϵ_0 = T(1000), ϵ_1 = zero(T), kwargs...) where {T <: Real, N}
    δ = zero(T)
    # ϵ = one(T)
    ϵ = ϵ_0
    noise = similar(C)
    noise .= T(0)
    noise[1] = T(1)
    r_list = T[]
    ϵ_list = T[]
    @show r_th = separable_ball_radius(T, lmo)
    res = (active_set = active_set, primal = zero(T))
    for _ in 1:certify_test
        C_rec = (1+ϵ) * C - ϵ * noise
        res = separable_distance(C_rec, lmo; active_set = res.active_set, kwargs...)
        δ = sqrt(res.primal)
        @show r = δ / ϵ
        if r < r_th
            println(r, " < ", r_th)
            return (sep = true, res = res)
        else
            ϵ = (ϵ + ϵ_1) / 2
            push!(r_list, r)
            push!(ϵ_list, ϵ)
        end
    end
    return (sep = false, res = (active_set = nothing, primal = zero(T)), r = r_list, ϵ = ϵ_list)
end
export separability_certification

function separability_certification(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}
    return separability_certification(correlation_tensor(ρ, lmo.dims), lmo; kwargs...)
end

function separable_ball_radius(::Type{T}, lmo::KSeparableLMO{T, N}) where {T <: Number, N}
    return maximum([separable_ball_radius(T, lmo.lmos[i].dims) for i in eachindex(lmo.lmos)])
end

function separable_ball_radius(::Type{T}, lmo::AlternatingSeparableLMO{T, N}) where {T <: Number, N}
    return separable_ball_radius(T, lmo.dims)
end