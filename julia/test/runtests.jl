# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test, SolpropJulia, LinearAlgebra
import FFTW
const S = SolpropJulia

function configuration(tasks)
    nw, nm, nt = 5, 2, 16
    # A constant linear problem with the same seven-stage interface; the source
    # vanishes, so the analytic exponential is independent of tableau coefficients.
    Dict(
        "linear" => fill(-0.1 + 0.2im, nw, nm),
        "prefactor" => zeros(ComplexF64,nw,nm),
        "synthesis" => Matrix{Float64}(I,nm,nm),
        "projection" => Matrix{Float64}(I,nm,nm),
        "projection_coarse" => Matrix{Float64}(I,nm,nm),
        "embedded" => false, "crop" => collect(0:nw-1),
        "mask" => [false,true,true,true,false], "norm_mask" => trues(nw),
        "weights" => ones(nw), "window" => ones(nt), "omega_window" => ones(nw),
        "rate" => zeros(nt), "coarse_rate" => zeros(2(nw-1)),
        "a" => zeros(7,7), "b" => zeros(7), "e" => zeros(7),
        "c" => collect(range(0,1;length=7)), "factor" => 0.0,
        "analytic" => false, "analytic_factor" => 0.0, "split" => false,
        "rtol" => 1e-8, "atol" => 1e-14,
        "threads" => tasks, "fft_threads" => 1, "blas_threads" => 1, "chunk" => 1,
    )
end

@testset "filtered prefix, reflected endpoints and overlapping halos" begin
    for nt in (9,256), half in (0,2,8)
        values=reshape([sin(0.2t)+0.01t for t in 1:2nt],nt,2)
        taps=[(-0.2)^k for k in 0:half]
        out=similar(values)
        for node in 1:2
            x=values[:,node]
            extended=vcat(2x[1].-x[half+1:-1:2],x,2x[end].-x[end-1:-1:end-half])
            smooth=[sum(taps[k+1]*(k==0 ? extended[t+half] :
                         extended[t+half-k]+extended[t+half+k]) for k in 0:half)
                    for t in 1:nt]
            expected=vcat(0.0,cumsum(0.03.*(smooth[1:end-1].+smooth[2:end])./2))
            S.integrate_column!(out,values,node,(taps=taps,dt=0.03))
            # Independent finite convolution and prefix summations: account for
            # filter width and every prefix addition, including cancellation.
            bound=128eps()*nt*(half+1)*norm(expected)
            @test norm(out[:,node]-expected) <= bound
            @test out[1,node] == 0
        end
    end
end

@testset "owned state, transforms, masks, rejection and task widths" begin
    for tasks in unique([1,Threads.nthreads()])
        engine = S.create(configuration(tasks))
        initial = complex.(reshape(collect(1.0:10.0),5,2), 0.3)
        S.load!(engine, initial)
        before = S.snapshot(engine)
        initial .= 0
        @test any(!iszero,before)
        @test all(iszero, before[[1,5],:])
        @test S.trial!(engine, 0.2) == 0
        @test S.snapshot(engine) == before
        # Discard the first candidate and try a different step length.
        @test S.trial!(engine, 0.1) == 0
        S.commit!(engine, 0.1)
        @test S.snapshot(engine) ≈ before .* exp.((-.1+.2im)*.1) rtol=8eps()
        @test engine.calls == 13
        @test sum(values(S.ledger(engine))) > 0
        fine = engine.work.fine
        fine .= 0
        fine[2:4,:] .= before[2:4,:]
        expected = copy(fine)
        mul!(engine.work.modal,engine.work.inverse_plan,fine)
        mul!(fine,engine.work.forward_plan,engine.work.modal)
        # Two length-16 FFTs: a normwise O(eps log N) backward-error budget.
        @test norm(fine-expected) <= 32eps()*log2(16)*norm(expected)
        S.load!(engine,zeros(ComplexF64,5,2))
        @test S.trial!(engine,0.1) == 0
        S.commit!(engine,0.1)
        @test all(iszero,S.snapshot(engine))
    end
end

@testset "exact zero support retains reflected and prefix conventions" begin
    for nt in (9,256), half in (0,2,8), interval in (1:0,1:1,nt:nt,1:nt,3:6)
        values=zeros(nt,1)
        for t in interval
            values[t,1]=cos(0.7t)
        end
        p=(taps=[(-0.2)^k for k in 0:half],dt=0.03)
        out=fill(NaN,nt,1)
        # Invoke the retained generic traversal as an independent witness to
        # support pruning, including exact zeros and both reflected endpoints.
        expected=similar(out)
        invoke(S.integrate_column!,Tuple{Any,Any,Any,Any},expected,values,1,p)
        S.integrate_column!(out,values,1,p)
        bound=128eps()*nt*(half+1)*max(norm(expected),floatmin(Float64))
        @test norm(out-expected) <= bound
        @test all(isfinite,out)
    end
end
