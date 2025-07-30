"""
    separable_distance(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); measure, fw_algorithm, kwargs...)
    separable_distance(C::Array{T, N}; matrix_basis, measure, fw_algorithm, kwargs...)

Computes the distance between the quantum density matrix `ρ` and the separable space under a specific `measure` via a specific `fw_algorithm`:
```
f(ρ) = min_{σ ∈ SEP} g(ρ,σ)
```
For the density matrix `ρ`, if the argument `dims` is omitted equally-sized subsystems are assumed, which is solving on the symmetry bipartite separable space.
For the correlation tensor `C`, if the argument `matrix_basis` is omitted, Gell-Mann matrix is assumed, which is Pauli basis for qubit systems.

The quantum state can also be given by a correlation tensor `C` corresponding to the experimental data from a set of (over-)completed `matrix_basis`.
If the argument `matrix_basis` is omitted the generalized Gell-Mann basis are assumed.

The `measure` g(ρ,σ) can be set as
- `"2-norm"`,
- [`"relative-entropy"`](https://arxiv.org/abs/quant-ph/9702027),
- [squared `"Bures metric"`](https://arxiv.org/abs/quant-ph/9707035).

The `fw_algorithm` can be used as
- `FrankWolfe.frank_wolfe`
- `FrankWolfe.lazified_conditional_gradient`
- `FrankWolfe.away_frank_wolfe`
- `FrankWolfe.blended_pairwise_conditional_gradient`

Returns a named tuple `(σ, v, primal)` with:
- `σ` the closest density matrix in the separable space
- `v` the closest pure separable state ket on the boundary of the separable space
- `primal` primal value f(x), the distance to the separable space
- `active_set` all the pure separable states, which combined to the closest separable state `σ`
- `lmo` the structure for related computation
"""
function separable_distance(ρ::AbstractMatrix{CT}, dims::NTuple{N, Int}, lmo::LMO=AlternatingSeparableLMO(float(real(CT)), dims); measure::String = "2-norm", fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {CT <: Number, N, LMO <: SeparableLMO}
    C = correlation_tensor(ρ, lmo.dims, lmo.matrix_basis)
    x, v, primal, noise_level, active_set, lmo = separable_distance(C, lmo; measure, fw_algorithm, kwargs...)
    return (σ = density_matrix(x, lmo.dims, lmo.matrix_basis), v = v, primal = primal / 2^(N-1), noise_level = noise_level, active_set = active_set, lmo = lmo)
end
export separable_distance

function separable_distance(C::Array{T, N}, matrix_basis::NTuple{N, Vector{<:AbstractMatrix{Complex{T}}}}, lmo::LMO=AlternatingSeparableLMO(T, isqrt.(size(C))); measure::String = "2-norm", fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {T <: Real, N, LMO <: SeparableLMO{T, N}}
    return separable_distance(C, lmo; measure, fw_algorithm, kwargs...)
end

function separable_distance(
    C::Array{T, N},
    lmo::SeparableLMO{T, N};
    noise_mixture::Bool = false,
    noise = nothing,
    noise_level::T = T(0),
    noise_atol::Real = 1e-3,
    noise_update_count::Int = 10,
    measure::String = "2-norm",
    fw_algorithm::Function = FrankWolfe.blended_pairwise_conditional_gradient,
    ini_sigma = Matrix{Complex{T}}(LA.I, prod(lmo.dims), prod(lmo.dims)) / prod(lmo.dims),
    ini_tensor = correlation_tensor(ini_sigma, lmo.dims),
    active_set = nothing,
    max_active = 10^4,
    epsilon = 1e-6,
    lazy = true,
    max_iteration = 10^5,
    verbose = 1,
    logfile = nothing,
    callback_iter = 10^4,
    shortcut = false, # primal > 10 dual_gap stopping criterion
    shortcut_scale = 10,
    kwargs...
) where {T <: Real, N}
    if verbose >0 
        if typeof(lmo) <: KSeparableLMO{T}
            lmo.lmos[1].parallelism && @info "The number of threads is $(Threads.nthreads())"
        else
            lmo.parallelism && @info "The number of threads is $(Threads.nthreads())"
        end
    end

    if isnothing(logfile)
        logfile = stdout
    else
        redirect_stdout(logfile)
    end
    # left for consistency between runs
    Random.seed!(0)
    
    if isnothing(noise)
        noise = similar(C)
        noise .= T(0)
        noise[1] = T(1)
    end
    rp = Ref(noise_level)
    dotCChalf = LA.dot(C, C) / 2
    dotIIhalf = LA.dot(noise, noise) / 2
    dotCI = LA.dot(C, noise)
    function f_noise(x, rp)
        return (1 - rp[])^2 * dotCChalf + rp[]^2 * dotIIhalf + rp[] * (1 - rp[]) * dotCI + LA.dot(x, x) / 2 - (1 - rp[])* LA.dot(C, x) - rp[] * LA.dot(noise, x)
    end
    function grad_noise!(storage, x, rp) # in-place gradient computation
        @. storage = x - ((1 - rp[]) * C + rp[] * noise)
    end
    f(x) = f_noise(x, rp)
    grad!(storage, x) = grad_noise!(storage, x, rp)

    if active_set === nothing
        x0 = FrankWolfe.compute_extreme_point(lmo, ini_tensor - C)
        active_set = FrankWolfe.ActiveSetQuadraticProductCaching([(one(T), x0)], LA.I, -C)
    elseif active_set isa FrankWolfe.ActiveSetQuadraticProductCaching
        FrankWolfe.update_active_set_quadratic!(active_set, -C)
        x0 = FrankWolfe.get_active_set_iterate(active_set)
    end

    trajectory_arr = []
    callback = build_callback(trajectory_arr, epsilon, max_active, shortcut, shortcut_scale, noise_mixture, rp, noise_atol, noise_update_count, C, noise, verbose, logfile, callback_iter)

    if fw_algorithm in [FrankWolfe.frank_wolfe, FrankWolfe.lazified_conditional_gradient]
        x, v, primal, dual_gap, traj_data = fw_algorithm(
            f,
            grad!,
            lmo,
            x0;
            line_search = FrankWolfe.Shortstep(one(T)),
            epsilon = zero(T), # avoid standard stopping criterion for the dual gap
            max_iteration,
            callback,
            verbose = false,
            kwargs...
        )
        active_set = FrankWolfe.ActiveSetQuadraticProductCaching([(one(T), x)], LA.I, -C)
    else
        x, v, primal, dual_gap, traj_data, active_set = fw_algorithm(
            f,
            grad!,
            lmo,
            active_set;
            line_search = FrankWolfe.Shortstep(one(T)),
            epsilon = zero(T), # avoid standard stopping criterion for the dual gap
            max_iteration,
            callback,
            verbose = false,
            lazy,
            kwargs...
        )
    end

    # print last iteration
    if verbose == 1 && noise_mixture == false
        Printf.@printf(
            stdout,
            "%s    %.4e    %.4e    %s\n",
            lpad("Last", 12),
            primal,
            dual_gap,
            lpad(length(active_set), 7)
            )
    elseif verbose == 1 && noise_mixture
        Printf.@printf(
            stdout,
            "%s    %.4e    %.4e    %.4e    %s\n",
            lpad("Last", 12),
            primal,
            dual_gap,
            rp[],
            lpad(length(active_set), 7)
            )
    elseif verbose == 2 && noise_mixture == false
        Printf.@printf(
            stdout,
            "%s    %.4e    %.4e    %.4e    %s   %s    %s\n",
            lpad("Last", 12),
            primal,
            dual_gap,
            lmo.fwdata.fw_time[1],
            lpad(Printf.@sprintf("%.4e", lmo.fwdata.fw_iter[1] / lmo.fwdata.fw_time[1]), 10),
            lpad(length(active_set), 7),
            lpad(lmo.fwdata.lmo_counts[1], 7)
            )
    elseif verbose == 2 && noise_mixture
        Printf.@printf(
            stdout,
            "%s    %.4e    %s    %.4e    %.4e    %.4e    %s   %s    %s\n",
            lpad("Last", 12),
            primal,
            lpad("------", 10),
            dual_gap,
            rp[],
            lmo.fwdata.fw_time[1],
            lpad(Printf.@sprintf("%.4e", lmo.fwdata.fw_iter[1] / lmo.fwdata.fw_time[1]), 10),
            lpad(length(active_set), 7),
            lpad(lmo.fwdata.lmo_counts[1], 7)
            )
    end
    if verbose > 0
        if length(active_set) > max_active
            @info "Stop: active set too large"
        elseif noise_mixture && rp[] > 1
            @info "Stop: noise fully added"
        elseif !noise_mixture && shortcut && primal / dual_gap > shortcut_scale
            @info "Stop: primal great larger than dual gap (shortcut)"
        elseif primal < epsilon
            @info "Stop: primal small enough"
        elseif lmo.fwdata.fw_iter[1] >= max_iteration
            @info "Stop: maximum iteration reached"
        else
            @info "Stop: unknown reason"
        end
    end
    flush(logfile)
    return (x = x, v = v, primal = primal, noise_level = rp[], active_set = active_set, lmo = lmo, traj_data = traj_data)
end
