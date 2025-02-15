using LinearAlgebra
using ComponentArrays
using Enzyme
using Random

# acts like a sparse matrix under my_rmul! (usually add method to LineraAlgebera)
struct Op2{T}
    i::Int
    j::Int
    a::T
    b::T
    c::T
end
_ckparams(C::Op2) = (C.a + C.b * im, C.a + C.c * im)
function my_rmul!(A::AbstractVecOrMat{Complex{T}}, C::Op2{T}) where {T}
    α, β = _ckparams(C)
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

function fe_rmul!(A::AbstractVecOrMat{Complex{T}}, C::Op2{T}) where {T}
    α, β = _ckparams(C)
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

""" Component Vector A holds values for many Op2
    where the ith Op2 is Op2(k, j, A.as[i], A.bs[i], A.cs[i])
    where k, j follow the sequence of KJIterator
"""
function init_A(lo)
    lA = lo * (lo-1)÷ 2
    ComponentArray(;as=rand(lA), bs=rand(lA), cs=rand(lA))
end

function rmul_from_A!(m, A)
    as, bs, cs = A.as, A.bs, A.cs
    for ((k, j, _), a, b, c) in zip(KJIterator(), as, bs, cs)
        my_rmul!(m, Op2(k, j, a, b, c))
    end
    nothing
end

function fe_rmul_from_A!(m, A)
    as, bs, cs = A.as, A.bs, A.cs
    foreach(zip(KJIterator(), as, bs, cs)) do ((k, j, _), a, b, c)
        fe_rmul!(m, Op2(k, j, a, b, c))
    end
    nothing
end

""" Component Vector AA holds values for a bunch of A
    where the ith A is AA[Symbol("d\$(i)")]
    e.g. 3rd A is given by AA.d3
"""
init_AA(lo, nd) = ComponentArray(; (Symbol("d$(i)")=>init_A(lo) for i in 1:nd)...)

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

""" Component Vector AAA holds values for a bunch of AA
    where the ith AA is AAASymbol("k\$(i)")]
    e.g. 3rd AA is given by AAA.k3
"""
init_AAA(los, nd) = ComponentArray(; (Symbol("k$(i)")=>init_AA(lo, nd) for (i, lo) in enumerate(los))...)

function each_m_to_AAmat!(M, AAA)
    for (i, vi) in enumerate(valkeys(AAA))
        m_to_AAmat!(M[i], view(AAA, vi))
    end
end

function fe_each_m_to_AAmat!(M, AAA)
    foreach(enumerate(valkeys(AAA))) do (i, vi)
        m_to_AAmat!(M[i], view(AAA, vi))
    end
end

# TODO: try foreach
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

@noinline function _obj_inner!(ik, bp, mats, storage)
    # Mismatched activity!
    # foreach(enumerate(bp)) do (ib, bk)
    #     diagprodthree!(storage.diags,
    #                    storage.M[ik],
    #                    mats[ik][ib],
    #                    adjoint(storage.M[bk]))
    # end
    for (ib, bk) in enumerate(bp)
        diagprodthree!(storage.diags,
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

function fe_objective!(AAA, C, storage)
    fe_each_m_to_AAmat!(storage.M, AAA)
    res = 0.0
    # Mismatched activity!
    # foreach(enumerate(C.bps)) do (ik, bp)
    #     fill!(storage.diags, 0.0)
    #     _obj_inner!(ik, bp, C.mats, storage)
    #     res += sum(x->1-abs2(x), storage.diags)
    # end
    for (ik, bp) in enumerate(C.bps)
        fill!(storage.diags, 0.0)
        _obj_inner!(ik, bp, C.mats, storage)
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

function init_primal(; seed, lnn, nk, nbs, nd, lovary)
    possible_lo = lnn .+ lovary
    Random.seed!(seed)
    los = rand(possible_lo, nk)
    bps = map(x->rand(eachindex(los), nbs), los)
    mats = init_mats(bps, los)

    (; AAA = init_AAA(los, nd),
     C = (; bps, mats),
     storage = init_storage(lnn, los))
end

function run_themrev!(time_dat; kwargs...)
    (;AAA, C, storage) = init_primal(; kwargs...)
    ∂AAA = Enzyme.make_zero(AAA)
    ∂storage = Enzyme.make_zero(storage)

    (;foreach_first, foreach_second, for_first, for_second) = time_dat

    push!(foreach_first.primal.comp, (@timed fe_objective!(AAA, C, storage)))
    push!(foreach_first.primal.run, (@timed fe_objective!(AAA, C, storage)))
    push!(foreach_first.grad.comp, (@timed ∇fe_objective!(∂AAA, AAA, C, storage, ∂storage)))
    push!(foreach_first.grad.run, (@timed ∇fe_objective!(∂AAA, AAA, C, storage, ∂storage)))

    push!(for_second.primal.comp, (@timed objective!(AAA, C, storage)))
    push!(for_second.primal.run, (@timed objective!(AAA, C, storage)))
    push!(for_second.grad.comp, (@timed ∇objective!(∂AAA, AAA, C, storage, ∂storage)))
    push!(for_second.grad.run, (@timed ∇objective!(∂AAA, AAA, C, storage, ∂storage)))

    nothing
end

function run_them!(time_dat; kwargs...)
    (;AAA, C, storage) = init_primal(;kwargs...)
    ∂AAA = Enzyme.make_zero(AAA)
    ∂storage = Enzyme.make_zero(storage)

    (;foreach_first, foreach_second, for_first, for_second) = time_dat

    push!(for_first.primal.comp, (@timed objective!(AAA, C, storage)))
    push!(for_first.primal.run, (@timed objective!(AAA, C, storage)))
    push!(for_first.grad.comp, (@timed ∇objective!(∂AAA, AAA, C, storage, ∂storage)))
    push!(for_first.grad.run, (@timed ∇objective!(∂AAA, AAA, C, storage, ∂storage)))

    push!(foreach_second.primal.comp, (@timed fe_objective!(AAA, C, storage)))
    push!(foreach_second.primal.run, (@timed fe_objective!(AAA, C, storage)))
    push!(foreach_second.grad.comp, (@timed ∇fe_objective!(∂AAA, AAA, C, storage, ∂storage)))
    push!(foreach_second.grad.run, (@timed ∇fe_objective!(∂AAA, AAA, C, storage, ∂storage)))

    nothing
end

TimedRes{T} = @NamedTuple{value::T,
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
    Base.time_print(Base.stdout, time*1e9, gcstats.allocd, gcstats.total_time,
                    Base.gc_alloc_count(gcstats), 0, 0, true; msg=msg)
end

fixKw(f; kwargs...) = (args...)->f(args...; kwargs...)
function show_all(time_dat)
    foreach(fixKw(show_one; msg="foreach, primal run"), time_dat.foreach_second.primal.run)
    foreach(fixKw(show_one; msg="for, primal run"), time_dat.for_second.primal.run)
    println()
    foreach(fixKw(show_one; msg="foreach, grad run"), time_dat.foreach_second.grad.run)
    foreach(fixKw(show_one; msg="for, grad run"), time_dat.for_second.grad.run)
    println()
    foreach(fixKw(show_one; msg="foreach before for, grad comp"), time_dat.foreach_first.grad.comp)
    foreach(fixKw(show_one; msg="for before foreach, grad comp"), time_dat.for_first.grad.comp)
    println()
    foreach(fixKw(show_one; msg="foreach after for, grad comp"), time_dat.foreach_second.grad.comp)
    foreach(fixKw(show_one; msg="for after foreach, grad comp"), time_dat.for_second.grad.comp)
    println()
end

function main()
    time_dat = (;foreach_first = PrimalGradRes{Float64, Nothing}(),
                foreach_second = PrimalGradRes{Float64, Nothing}(),
                for_first = PrimalGradRes{Float64, Nothing}(),
                for_second = PrimalGradRes{Float64, Nothing}())
    for lnn in 31:2:39
        run_them!(time_dat; seed = 123, lnn = lnn, nk = 20, nbs = 5, nd = 5, lovary = 0:9)
        run_themrev!(time_dat; seed = 123, lnn = lnn+1, nk = 20, nbs = 5, nd = 5, lovary = 0:9)
    end
    show_all(time_dat)
    time_dat
end

#main()
