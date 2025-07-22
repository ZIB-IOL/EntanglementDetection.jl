function build_callback(trajectory_arr, epsilon, max_active, shortcut, shortcut_scale, noise_mixture, rp, Δp, noise_k, C, Id, verbose, logfile, callback_iter)
    primal_prev = Inf
    noise_print_update = true
    noise_print_count = 0
    noise_update_count = 0
    if verbose == 1 && noise_mixture == false
        Printf.@printf(
            stdout,
            "%s  %s  %s    %s\n",
            lpad("Iteration", 12),
            lpad("Primal", 12),
            lpad("Dual gap", 10),
            lpad("#Atoms", 7),
            )
    elseif verbose == 1 && noise_mixture
        Printf.@printf(
            stdout,
            "%s  %s  %s    %s    %s\n",
            lpad("Iteration", 12),
            lpad("Primal", 12),
            lpad("Dual gap", 10),
            lpad("Noise", 10),
            lpad("#Atoms", 7),
            )
    elseif verbose == 2 && noise_mixture == false
        Printf.@printf(
            stdout,
            "%s  %s  %s    %s   %s    %s    %s\n",
            lpad("Iteration", 12),
            lpad("Primal", 12),
            lpad("Dual gap", 10),
            lpad("Time (sec)", 10),
            lpad("#It/sec", 10),
            lpad("#Atoms", 7),
            lpad("#LMOs", 7)
            )
    elseif verbose == 2 && noise_mixture
        Printf.@printf(
            stdout,
            "%s  %s  %s    %s    %s    %s   %s    %s    %s\n",
            lpad("Iteration", 12),
            lpad("Primal", 12),
            lpad("Primal_prev", 12),
            lpad("Dual gap", 10),
            lpad("Noise", 10),
            lpad("Time (sec)", 10),
            lpad("#It/sec", 10),
            lpad("#Atoms", 7),
            lpad("#LMOs", 7)
            )
    end
    flush(logfile)
    function callback(state, args...)
        if length(args) > 0
            active_set = args[1]
            # push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set), rp[]))
        else
            active_set = []
            # push!(trajectory_arr, (FrankWolfe.callback_state(state)..., rp[]))
        end
        state.lmo.fwdata.fw_iter[1] = state.t
        state.lmo.fwdata.fw_time[1] = state.time

        if (mod(state.t, callback_iter) == 0 || noise_print_update)
            if verbose == 1 && noise_mixture == false
                Printf.@printf(
                    stdout,
                    "%s    %.4e    %.4e    %s\n",
                    lpad(state.t, 12),
                    state.primal,
                    state.dual_gap,
                    lpad(length(active_set), 7)
                    )
            elseif verbose == 1 && noise_mixture
                Printf.@printf(
                    stdout,
                    "%s    %.4e    %.4e    %.4e    %s\n",
                    lpad(state.t, 12),
                    state.primal,
                    state.dual_gap,
                    rp[],
                    lpad(length(active_set), 7)
                    )
            elseif verbose == 2 && noise_mixture == false
                Printf.@printf(
                    stdout,
                    "%s    %.4e    %.4e    %.4e    %s   %s    %s\n",
                    lpad(state.t, 12),
                    state.primal,
                    state.dual_gap,
                    state.time,
                    lpad(Printf.@sprintf("%.4e", state.t / state.time), 10),
                    lpad(length(active_set), 7),
                    lpad(state.lmo.fwdata.lmo_counts[1], 7)
                    )
            elseif verbose == 2 && noise_mixture
                Printf.@printf(
                    stdout,
                    "%s    %.4e    %s    %.4e    %.4e    %.4e    %s   %s    %s\n",
                    lpad(state.t, 12),
                    state.primal,
                    lpad(Printf.@sprintf("%.4e", primal_prev), 10),
                    state.dual_gap,
                    rp[],
                    state.time,
                    lpad(Printf.@sprintf("%.4e", state.t / state.time), 10),
                    lpad(length(active_set), 7),
                    lpad(state.lmo.fwdata.lmo_counts[1], 7)
                    )
            end
            flush(logfile)
        end
        noise_print_update = false

        if length(active_set) > max_active
            verbose > 0 && @info "active set is too large"
            return false
        end

        if noise_mixture
            if rp[] > 1 # stop if the noise is fully added
                verbose > 0 && @info "noise is fully added"
                return false
            end
            noise_update_count += 1
            if state.t > 100 && noise_update_count < noise_k
                return true # do not update the noise
            end
            noise_update_count = 0
            if state.primal < primal_prev && state.primal / state.dual_gap > 1 + state.dual_gap * 10^4 && state.primal / state.dual_gap > shortcut_scale # update the noise
            noise_print_count += 1
                if noise_print_count > (1 / Δp) ÷ 10
                    noise_print_update = true
                    noise_print_count = 0
                end
                rp[] += Δp
                primal_prev = state.primal
                if length(args) > 0
                    if typeof(active_set) == FrankWolfe.ActiveSetQuadraticLinearSolve
                        FrankWolfe.update_active_set_quadratic!(active_set.active_set, -((1 - rp[]) * C + rp[] * Id))
                    elseif typeof(active_set) == FrankWolfe.ActiveSetQuadraticProductCaching
                        FrankWolfe.update_active_set_quadratic!(active_set, -((1 - rp[]) * C + rp[] * Id))
                    end
                end
                return true
            end
        end

        if !noise_mixture && shortcut && state.primal / state.dual_gap > shortcut_scale # when gap is large enough -> entangled, stop. (remove if we not use it)
            verbose > 0 && @info "shortcut"
            return false
        end

        if state.primal < epsilon # stop if the primal is small enough (main stopping criterion)
            verbose > 0 && @info "primal is small enough"
            return false
        end

        return true # control when to stop
    end
    return callback
end
