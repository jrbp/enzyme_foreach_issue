using LinearAlgebra
using ComponentArrays
using Enzyme
using Random
import Pkg

""" For in-place right mulitplying with matrices via
    my_rmul! or fe_my_rmul! (acts like a particular sparse matrix)
"""
struct Op2{T}
    i::Int
    j::Int
    a::T
    b::T
    c::T
end

""" Component Vector A holds values for many Op2
    where the ith Op2 is Op2(k, j, A.as[i], A.bs[i], A.cs[i])
    where k, j follow the sequence of KJIterator
"""
function init_A(lo)
    lA = (lo * (lo-1)) ÷ 2
    ComponentArray(;as=rand(lA), bs=rand(lA), cs=rand(lA))
end

""" Component Vector AA holds values for a bunch of A
    where the ith A is AA[Symbol("d\$(i)")]
    e.g. 3rd A is given by AA.d3
"""
init_AA(lo, nd) = ComponentArray(; (Symbol("d$(i)")=>init_A(lo) for i in 1:nd)...)

""" Component Vector AAA holds values for a bunch of AA
    where the ith AA is AAA[Symbol("k\$(i)")]
    e.g. 3rd AA is given by AAA.k3
"""
init_AAA(los, nd) = ComponentArray(; (Symbol("k$(i)")=>init_AA(lo, nd) for (i, lo) in enumerate(los))...)

function my_rmul!(A::AbstractVecOrMat{Complex{T}}, C::Op2{T}) where {T}
    α, β = (C.a + C.b * im, C.a + C.c * im)
    Base.require_one_based_indexing(A)
    m, n = size(A, 1), size(A, 2)
    (C.j > n) && throw(DimensionMismatch("no"))
    for i in 1:m
        a1, a2 = A[i, C.i], A[i, C.j]
        A[i, C.i] = a1 * α - a2 * conj(β)
        A[i, C.j] = a1 * β + a2 * conj(α)
    end
    A
end

function fe_my_rmul!(A::AbstractVecOrMat{Complex{T}}, C::Op2{T}) where {T}
    α, β = (C.a + C.b * im, C.a + C.c * im)
    Base.require_one_based_indexing(A)
    m, n = size(A, 1), size(A, 2)
    (C.j > n) && throw(DimensionMismatch("no"))
    foreach(1:m) do i
        a1, a2 = A[i, C.i], A[i, C.j]
        A[i, C.i] = a1 * α - a2 * conj(β)
        A[i, C.j] = a1 * β + a2 * conj(α)
    end
    A
end

struct KJIterator end
_nextkj((k, j, c)) = c ? (1, j+1, false) : (k+1, j, j==k+1+1)
Base.iterate(::KJIterator, state=(1, 2, true)) = state, _nextkj(state)
Base.IteratorSize(::Type{KJIterator}) = Base.IsInfinite()

struct fe_KJIterator end
fe_nextkj((k, j, c)) = c ? (1, j+1, false) : (k+1, j, j==k+1+1)
Base.iterate(::fe_KJIterator, state=(1, 2, true)) = state, fe_nextkj(state)
Base.IteratorSize(::Type{fe_KJIterator}) = Base.IsInfinite()

function rmul_from_A!(m, A)
    as, bs, cs = A.as, A.bs, A.cs
    for ((k, j, _), a, b, c) in zip(KJIterator(), as, bs, cs)
        my_rmul!(m, Op2(k, j, a, b, c))
    end
    nothing
end

function fe_rmul_from_A!(m, A)
    as, bs, cs = A.as, A.bs, A.cs
    foreach(zip(fe_KJIterator(), as, bs, cs)) do ((k, j, _), a, b, c)
        fe_my_rmul!(m, Op2(k, j, a, b, c))
    end
    nothing
end

function rmul_from_AA!(m, AA)
    for x in valkeys(AA)
        rmul_from_A!(m, view(AA, x))
    end
end
function fe_rmul_from_AA!(m, AA)
    foreach(valkeys(AA)) do x
        fe_rmul_from_A!(m, view(AA, x))
    end
end

function m_to_AAmat!(m, AA)
    copyto!(m, I)
    rmul_from_AA!(m, AA)
end

function fe_m_to_AAmat!(m, AA)
    copyto!(m, I)
    fe_rmul_from_AA!(m, AA)
end

function each_m_to_AAmat!(M, AAA)
    for (i, vi) in enumerate(valkeys(AAA))
        m_to_AAmat!(M[i], view(AAA, vi))
    end
end

function fe_each_m_to_AAmat!(M, AAA)
    foreach(enumerate(valkeys(AAA))) do (i, vi)
        fe_m_to_AAmat!(M[i], view(AAA, vi))
    end
end

function diagprodthree!(res, A, B, C)
    for i in axes(A, 1)
        for j in axes(B, 1)
            for k in axes(C, 1)
                res[i] += A[i, j] * B[j, k] * C[k, i]
            end
        end
    end
    nothing
end

function truefe_diagprodthree!(res, A, B, C)
    # Mismatched activity!
    # will error if any single loop is changed
    foreach(axes(A, 1)) do i
        foreach(axes(B, 1)) do j
            foreach(axes(C, 1)) do k
                res[i] += A[i, j] * B[j, k] * C[k, i]
            end
        end
    end
end

function fe_diagprodthree!(res, A, B, C)
    for i in axes(A, 1)
        for j in axes(B, 1)
            for k in axes(C, 1)
                res[i] += A[i, j] * B[j, k] * C[k, i]
            end
        end
    end
    nothing
end

function _obj_inner!(ik, bp, mats, storage)
    for (ib, bk) in enumerate(bp)
        diagprodthree!(storage.diags,
                       storage.M[ik],
                       mats[ik][ib],
                       adjoint(storage.M[bk]))
    end
end

function truefe_obj_inner!(ik, bp, mats, storage)
    # Mismatched activity!
    foreach(enumerate(bp)) do (ib, bk)
        fe_diagprodthree!(storage.diags,
                       storage.M[ik],
                       mats[ik][ib],
                       adjoint(storage.M[bk]))
    end
end

function fe_obj_inner!(ik, bp, mats, storage)
    for (ib, bk) in enumerate(bp)
        fe_diagprodthree!(storage.diags,
                       storage.M[ik],
                       mats[ik][ib],
                       adjoint(storage.M[bk]))
    end
end

function objective!(AAA, C, storage)
    each_m_to_AAmat!(storage.M, AAA)
    res = 0.0
    for (ik, bp) in enumerate(C.bps)
        fill!(storage.diags, 0.0)
        _obj_inner!(ik, bp, C.mats, storage)
        res += sum(x->1-abs2(x), storage.diags)
    end
    res
end

function ∇objective!(∂AAA, AAA, C, storage, ∂storage)
    Enzyme.make_zero!(∂AAA)
    Enzyme.make_zero!(∂storage)
    Enzyme.autodiff(Enzyme.Reverse,
        objective!, Enzyme.Active,
        Enzyme.Duplicated(AAA, ∂AAA),
        Enzyme.Const(C),
        Enzyme.Duplicated(storage, ∂storage))
    nothing
end

function truefe_objective!(AAA, C, storage)
    fe_each_m_to_AAmat!(storage.M, AAA)
    # ## Mismatched activity! + doubles primal allocations unles res is Ref
    res = 0.0
    foreach(enumerate(C.bps)) do (ik, bp)
        fill!(storage.diags, 0.0)
        fe_obj_inner!(ik, bp, C.mats, storage)
        res += sum(x->1-abs2(x), storage.diags)
    end
    res
end

function ∇truefe_objective!(∂AAA, AAA, C, storage, ∂storage)
    Enzyme.make_zero!(∂AAA)
    Enzyme.make_zero!(∂storage)
    Enzyme.autodiff(Enzyme.Reverse,
        truefe_objective!, Enzyme.Active,
        Enzyme.Duplicated(AAA, ∂AAA),
        Enzyme.Const(C),
        Enzyme.Duplicated(storage, ∂storage))
    nothing
end

function sumfe_objective!(AAA, C, storage)
    fe_each_m_to_AAmat!(storage.M, AAA)
    ## Mismatched activity!
    sum(enumerate(C.bps)) do (ik, bp)
        fill!(storage.diags, 0.0)
        truefe_obj_inner!(ik, bp, C.mats, storage)
        sum(x->1-abs2(x), storage.diags)
    end
end

function ∇sumfe_objective!(∂AAA, AAA, C, storage, ∂storage)
    Enzyme.make_zero!(∂AAA)
    Enzyme.make_zero!(∂storage)
    Enzyme.autodiff(Enzyme.Reverse,
        sumfe_objective!, Enzyme.Active,
        Enzyme.Duplicated(AAA, ∂AAA),
        Enzyme.Const(C),
        Enzyme.Duplicated(storage, ∂storage))
    nothing
end

function fe_objective!(AAA, C, storage)
    fe_each_m_to_AAmat!(storage.M, AAA)
    res = 0.0
    for (ik, bp) in enumerate(C.bps)
        fill!(storage.diags, 0.0)
        fe_obj_inner!(ik, bp, C.mats, storage)
        res += sum(x->1-abs2(x), storage.diags)
    end
    res
end

function ∇fe_objective!(∂AAA, AAA, C, storage, ∂storage)
    Enzyme.make_zero!(∂AAA)
    Enzyme.make_zero!(∂storage)
    Enzyme.autodiff(Enzyme.Reverse,
        fe_objective!, Enzyme.Active,
        Enzyme.Duplicated(AAA, ∂AAA),
        Enzyme.Const(C),
        Enzyme.Duplicated(storage, ∂storage))
    nothing
end

function init_storage(lnn, los, T=Float64)
    M = map(lo->Matrix{Complex{T}}(undef, lnn, lo), los)
    diags = zeros(Complex{T}, lnn)
    (; M, diags)
end

function init_mats(bps, los, T=Float64)
    map(zip(los, bps)) do (lo, bp)
        map(bp) do (b)
            rand(Complex{T}, lo, los[b])
        end
    end
end

function init_params(; seed, lnn, nk, nbs, nd, lovary)
    possible_lo = lnn .+ lovary
    Random.seed!(seed)
    los = rand(possible_lo, nk)
    bps = map(x->rand(eachindex(los), nbs), los)
    mats = init_mats(bps, los)

    AAA = init_AAA(los, nd)
    ∂AAA = Enzyme.make_zero(AAA)
    C = (; bps, mats)
    storage = init_storage(lnn, los)
    ∂storage = Enzyme.make_zero(storage)

    (;∂AAA, AAA, C, storage, ∂storage)
end

function run_complicated!(time_dat; kwargs...)
    (; ∂AAA, AAA, C, storage, ∂storage) = init_params(; kwargs...)

    (;foreach_times, for_times) = time_dat

    push!(for_times.primal.comp, (@timed objective!(AAA, C, storage)))
    push!(for_times.primal.run, (@timed objective!(AAA, C, storage)))
    try
        push!(for_times.grad.comp, (@timed ∇objective!(∂AAA, AAA, C, storage, ∂storage)))
        push!(for_times.grad.run, (@timed ∇objective!(∂AAA, AAA, C, storage, ∂storage)))
    catch error
        println("for, gradient failed for $(kwargs) with $(error)")
    end

    push!(foreach_times.primal.comp, (@timed fe_objective!(AAA, C, storage)))
    push!(foreach_times.primal.run, (@timed fe_objective!(AAA, C, storage)))
    try
        push!(foreach_times.grad.comp, (@timed ∇fe_objective!(∂AAA, AAA, C, storage, ∂storage)))
        push!(foreach_times.grad.run, (@timed ∇fe_objective!(∂AAA, AAA, C, storage, ∂storage)))
    catch error
        println("for, gradient failed for $(kwargs) with $(error)")
    end

    nothing
end

function run_failing(; kwargs...)
    (; ∂AAA, AAA, C, storage, ∂storage) = init_params(; kwargs...)

    println("foreach/sum cases which fail:")
    @time "all foreach primal" truefe_objective!(AAA, C, storage)
    @time "all foreach primal" truefe_objective!(AAA, C, storage)
    try
        @time "all foreach grad" ∇truefe_objective!(∂AAA, AAA, C, storage, ∂storage)
        @time "all foreach grad" ∇truefe_objective!(∂AAA, AAA, C, storage, ∂storage)
    catch error
        println("all foreach , gradient failed for $(kwargs) with $(error)")
        println("all foreach , gradient failed for $(kwargs) with $(error)")
    end
    @time "sum + rest foreach primal" sumfe_objective!(AAA, C, storage)
    try
        @time "sum + rest foreach grad" ∇sumfe_objective!(∂AAA, AAA, C, storage, ∂storage)
        @time "sum + rest foreach grad" ∇sumfe_objective!(∂AAA, AAA, C, storage, ∂storage)
    catch error
        println("sum + rest foreach, gradient failed for $(kwargs) with $(error)")
    end
end


# Julia version 1.10 vs 1.11 have different time / timed macro behavior
TimedRes{T} = VERSION.minor == UInt32(11) ? @NamedTuple{ # Julia 1.11
    value::T,
    time::Float64,
    bytes::Int64,
    gctime::Float64,
    gcstats::Base.GC_Diff,
    lock_conflicts::Int64,
    compile_time::Float64,
    recompile_time::Float64} : @NamedTuple{ # Julia 1.10
        value::T,
        time::Float64,
        bytes::Int64,
        gctime::Float64,
        gcstats::Base.GC_Diff}

CompRunRes{T} = @NamedTuple{comp::TimedRes{T}, run::TimedRes{T}}
CompRunRes{T}() where {T} = (;comp=TimedRes{T}[], run=TimedRes{T}[])
PrimalGradRes{T, TG} = @NamedTuple{primal::CompRunRes{T}, grad::CompRunRes{TG}}
PrimalGradRes{T, TG}() where {T, TG} = (;primal=CompRunRes{T}(), grad=CompRunRes{TG}())

function show_one(timed_res; msg=nothing)
    (; time, gcstats) = timed_res
    (VERSION.minor == UInt32(11) ? Base.time_print(
        Base.stdout, time*1e9, gcstats.allocd, gcstats.total_time,
        Base.gc_alloc_count(gcstats), timed_res.lock_conflicts, timed_res.compile_time * 1e9, timed_res.recompile_time*1e9, true;
        msg=msg) : Base.time_print(
            Base.stdout, time*1e9, gcstats.allocd, gcstats.total_time,
            Base.gc_alloc_count(gcstats), 0, 0, true; msg=msg))
end

function show_all(time_dat)
    println("Complicated cases which succeeded")
    println("Here foreach is only used where it does not cause activity errors")
    println("for, primal run")
    foreach(show_one, time_dat.for_times.primal.run)
    println("foreach, primal run")
    foreach(show_one, time_dat.foreach_times.primal.run)
    println()
    println("for, grad run")
    foreach(show_one, time_dat.for_times.grad.run)
    println("foreach, grad run")
    foreach(show_one, time_dat.foreach_times.grad.run)
    println()
    println("for, primal comp")
    foreach(show_one, time_dat.for_times.primal.comp)
    println("foreach, primal comp")
    foreach(show_one, time_dat.foreach_times.primal.comp)
    println()
    println("note: first gradient comp always takes longer")
    println("for, grad comp")
    foreach(show_one, time_dat.for_times.grad.comp)
    println("foreach, grad comp")
    foreach(show_one, time_dat.foreach_times.grad.comp)
    println()
end


### Simple case which don't seem to have issues with foreach

function simpleobjective!(A)
    res = 0.0
    for (i, v) in enumerate(A)
        res += 1 - abs2(v)*i
    end
    res
end

function ∇simpleobjective!(∂A, A)
    Enzyme.make_zero!(∂A)
    Enzyme.autodiff(Enzyme.Reverse,
        simpleobjective!, Enzyme.Active,
        Enzyme.Duplicated(A, ∂A))
    nothing
end

function fe_simpleobjective!(A)
    res = 0.0
    foreach(enumerate(A)) do (i, v)
        res += 1 - abs2(v) * i
    end
    res
end

function ∇fe_simpleobjective!(∂A, A)
    Enzyme.make_zero!(∂A)
    Enzyme.autodiff(Enzyme.Reverse,
        fe_simpleobjective!, Enzyme.Active,
        Enzyme.Duplicated(A, ∂A))
    nothing
end

function sum_simpleobjective!(A)
    sum(enumerate(A)) do (i,v)
        1 - abs2(v) * i
    end
end

function ∇sum_simpleobjective!(∂A, A)
    Enzyme.make_zero!(∂A)
    Enzyme.autodiff(Enzyme.Reverse,
        sum_simpleobjective!, Enzyme.Active,
        Enzyme.Duplicated(A, ∂A))
    nothing
end

function run_simple()
    simple_vec = rand(10)
    ∂simple_vec = Enzyme.make_zero(simple_vec)
    println("Simple foreach/sum cases which work:")
    @time  "simple run  for    " simpleobjective!(simple_vec)
    @time  "simple run  for    " simpleobjective!(simple_vec)
    @time  "simple run  foreach" fe_simpleobjective!(simple_vec)
    @time  "simple run  foreach" fe_simpleobjective!(simple_vec)
    @time  "simple run  sum    " sum_simpleobjective!(simple_vec)
    @time  "simple run  sum    " sum_simpleobjective!(simple_vec)
    @time  "simple grad for    " ∇simpleobjective!(∂simple_vec, simple_vec)
    @time  "simple grad for    " ∇simpleobjective!(∂simple_vec, simple_vec)
    @time  "simple grad foreach" ∇fe_simpleobjective!(∂simple_vec, simple_vec)
    @time  "simple grad foreach" ∇fe_simpleobjective!(∂simple_vec, simple_vec)
    @time  "simple grad sum    " ∇sum_simpleobjective!(∂simple_vec, simple_vec)
    @time  "simple grad sum    " ∇sum_simpleobjective!(∂simple_vec, simple_vec)
end

function main(lnns=15:20)
    println("Julia: v$(VERSION)")
    Pkg.status()
    println()

    time_dat = (;foreach_times = PrimalGradRes{Float64, Nothing}(),
                for_times = PrimalGradRes{Float64, Nothing}(),)
    foreach(lnns) do lnn
        println(lnn)
        run_complicated!(time_dat; seed = 123, lnn = lnn, nk = 20, nbs = 5, nd = 5, lovary = 0:9)
    end
    show_all(time_dat)
    println()

    run_simple()
    println()

    run_failing(; seed = 123, lnn = 3, nk = 20, nbs = 5, nd = 5, lovary = 0:9)
    time_dat
end

main()
