function threaded_foreach(f, ntasks::Int)
    nt = Threads.nthreads()
    Threads.@threads for tid in 1:nt
        for task in tid:nt:ntasks
            f(tid, task)
        end
    end
end

function AlternatingCore(work::Workspace{T, N}, lmo::AlternatingSeparableLMO{T, N}, dir::AbstractArray{T, N}) where {T <: Real, N}
    for n in 1:N
        Random.randn!(work.pure_kets[n])
        LA.normalize!(work.pure_kets[n])
        _correlation_tensor_ket!(work.pure_tensors[n], work.pure_kets[n], lmo.matrix_basis[n])
    end

    obj = typemax(T)
    obj_last = typemax(T)
    tensors = ntuple(n -> Vector{T}(undef, lmo.dims[n]^2), Val(N))
    for _ in 1:lmo.max_iter
        obj = zero(T)
        for n in 1:N
            _reduced_tensor!(work.reduced_tensors[n], work.pure_tensors, dir, n, work.reduced_setdiffs[n])
            _density_matrix!(work.reduced_matrices[n], work.reduced_tensors[n], lmo.matrix_basis[n])
            obj = _eigmin!(work.pure_kets[n], work.reduced_matrices[n], work.blas_workspaces[n])
            _correlation_tensor_ket!(work.pure_tensors[n], work.pure_kets[n], lmo.matrix_basis[n])
        end

        if obj_last - obj ≤ lmo.threshold
            break
        end

        obj_last = obj
    end
    for n in 1:N
        tensors[n] .= work.pure_tensors[n]
    end
    return (tensors = tensors, obj = obj)
end

function FrankWolfe.compute_extreme_point(lmo::AlternatingSeparableLMO{T, N}, dir::AbstractArray{T, N}; kwargs...) where {T <: Real, N}
    lmo.fwdata.lmo_counts[1] += 1
    if lmo.parallelism
        tensors = [ntuple(n -> Vector{T}(undef, lmo.dims[n]^2), Val(N)) for _ in 1:lmo.nb]
        objs = [typemax(T) for _ in 1:lmo.nb]
        threaded_foreach(lmo.nb) do tid, task
            tensors[task], objs[task] = AlternatingCore(lmo.workspaces[tid], lmo, dir)
        end
        idx = argmin(objs) # find the best pure state
        return PureState{T, N, typeof(lmo)}(tensors[idx], objs[idx], lmo)
    else
        best_obj = typemax(T)
        best_pure_tensors = ntuple(n -> Vector{T}(undef, lmo.dims[n]^2), Val(N))
        for _ in 1:lmo.nb
            tensor, obj = AlternatingCore(lmo.workspaces[1], lmo, dir)
            if obj < best_obj
                best_obj = obj
                for n in 1:N
                    best_pure_tensors[n] .= tensor[n]
                end
            end
        end
        return PureState{T, N, typeof(lmo)}(best_pure_tensors, best_obj, lmo)
    end
end

function FrankWolfe.compute_extreme_point(lmo::EnumeratingSeparableLMO{T, N}, dir::AbstractArray{T, N}; kwargs...) where {T <: Real, N}
    work = lmo.workspace
    best_obj = typemax(T)
    best_pure_tensors = ntuple(n -> Vector{T}(undef, lmo.dims[n]^2), Val(N))
    for ci in CartesianIndices(length.(lmo.approximations))
        for n in 2:N
            _populate!(work.pure_kets[n], lmo.approximations[n-1], ci[n-1])
            # convert to tensor
            _correlation_tensor_ket!(work.pure_tensors[n], work.pure_kets[n], lmo.matrix_basis[n])
        end
        # computes the reduced tensor for the first party
        _reduced_tensor!(work.reduced_tensors[1], work.pure_tensors, dir, 1, work.reduced_setdiffs[1])
        # convert it to a density matrix
        _density_matrix!(work.reduced_matrices[1], work.reduced_tensors[1], lmo.matrix_basis[1])
        # the optimal state is the solution of an eigenvalue problem; modify pure_kets[1] in place
        obj = _eigmin!(work.pure_kets[1], work.reduced_matrices[1], work.blas_workspaces[1])
        if obj < best_obj
            best_obj = obj
            # update pure_tensors[1]
            _correlation_tensor_ket!(work.pure_tensors[1], work.pure_kets[1], lmo.matrix_basis[1])
            for n in 1:N
                best_pure_tensors[n] .= work.pure_tensors[n]
            end
        end
    end
    lmo.fwdata.lmo_counts[1] += 1
    return PureState{T, N, typeof(lmo)}(best_pure_tensors, best_obj, lmo)
end

function FrankWolfe.compute_extreme_point(lmo::KSeparableLMO{T, N}, dir::Array{T, N}; kwargs...) where {T <: Real, N}
    best_obj = typemax(T)
    best_x = similar(dir)
    for i in eachindex(lmo.lmos)
        x = FrankWolfe.compute_extreme_point(lmo.lmos[i], group_dims(dir, lmo.lmos[i].dims, lmo.partitions[i]); kwargs...)
        if x.obj < best_obj
            best_obj = x.obj
            best_x .= ungroup_dims(x, lmo.dims, lmo.partitions[i])
        end
    end
    return best_x
end

function FrankWolfe.muladd_memory_mode(memory_mode::FrankWolfe.InplaceEmphasis, d::Array{T, N}, x::AbstractArray{T, N}, v::PureState{T, N}) where {T <: Real, N}
    @inbounds for i in eachindex(v)
        d[i] = x[i] - v[i]
    end
    return d
end

function FrankWolfe.active_set_update_iterate_pairwise!(x::Array{T, N}, lambda::Real, fw_atom::PureState{T, N}, away_atom::PureState{T, N}) where {T <: Real, N}
    @inbounds for i in eachindex(fw_atom)
        x[i] += lambda * (fw_atom[i] - away_atom[i])
    end
    return x
end

function FrankWolfe.compute_active_set_iterate!(active_set::FrankWolfe.ActiveSetQuadraticProductCaching{AT, T, IT}) where {IT <: Array{T, N}, AT <: PureState{T, N}} where {T <: Real, N}
    active_set.x .= zero(T)
    for (λi, ai) in active_set
        @inbounds for x in eachindex(ai)
            active_set.x[x] += λi * ai[x]
        end
    end
    return active_set.x
end

function FrankWolfe.active_set_update_scale!(x::IT, lambda, atom::AT) where {IT <: Array{T, N}, AT <: PureState{T, N}} where {T <: Real, N}
    @inbounds for i in eachindex(atom)
        x[i] = x[i] * (1 - lambda) + lambda * atom[i]
    end
    return x
end
