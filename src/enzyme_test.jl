using LinearAlgebra
using ComponentArrays
using Enzyme

# acts like a sparse matrix under rmul!
struct Op2{T}
    i::Int
    j::Int
    a::T
    b::T
    c::T
end
_ckparams(C::Op2) = (C.a + C.b * im, C.a + C.c * im)
function LinearAlgebra.rmul!(A::AbstractVecOrMat{Complex{T}}, C::Op2{T}) where {T}
    α, β = _ckparams(C)
    Base.require_one_based_indexing(A)
    m, n = size(A, 1), size(A, 2)
    (C.j > n) && throw(DimensionMismatch("no"))
    foreach(1:m) do i
        a1, a2 = A[i, C.i], A[i, C.j]
        A[i, C.i] = a1 * α - a2 * conj(β)
        A[i, C.j] = a1 * β + a2 * conj(α)
    end
    # for i in 1:m
    #     a1, a2 = A[i, C.i], A[i, C.j]
    #     A[i, C.i] = a1 * α - a2 * conj(β)
    #     A[i, C.j] = a1 * β + a2 * conj(α)
    # end
    A
end

struct KJIterator end
_nextkj((k, j, c)) = c ? (1, j+1, false) : (k+1, j, j==k+1+1)
Base.iterate(iter::KJIterator, state=(1, 2, true)) = state, _nextkj(state)
Base.IteratorSize(::Type{KJIterator}) = Base.IsInfinite()

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
    foreach(zip(KJIterator(), as, bs, cs)) do ((k, j, _), a, b, c)
        rmul!(m, Op2(k, j, a, b, c))
    end
    # for ((k, j, _), a, b, c) in zip(KJIterator(), as, bs, cs)
    #     rmul!(m, Op2(k, j, a, b, c))
    # end
    nothing
end

""" Component Vector AA holds values for a bunch of A
    where the ith A is AA[Symbol("d\$(i)")]
    e.g. 3rd A is given by AA.d3
"""
init_AA(lo, nd) = ComponentArray(; (Symbol("d$(i)")=>init_A(lo) for i in 1:nd)...)

function rmul_from_AA!(m, AA)
    foreach(valkeys(AA)) do x
        rmul_from_A!(m, view(AA, x))
    end
    # for x in valkeys(AA)
    #     rmul_from_A!(m, view(AA, x))
    # end
end

function m_to_AAmat!(m, AA)
    copyto!(m, I)
    rmul_from_AA!(m, AA)
end

""" Component Vector AAA holds values for a bunch of AA
    where the ith AA is AAA[Symbol("k\$(i)")]
    e.g. 3rd AA is given by AAA.k3
"""
init_AAA(los, nd) = ComponentArray(; (Symbol("k$(i)")=>init_AA(lo, nd) for (i, lo) in enumerate(los))...)

function each_m_to_AAmat!(M, AAA)
    foreach(enumerate(valkeys(AAA))) do (i, vi)
        m_to_AAmat!(M[i], view(AAA, vi))
    end
    # for (i, vi) in enumerate(valkeys(AAA))
    #     m_to_AAmat!(M[i], view(AAA, vi))
    # end
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

# function _obj_inner1!(ik, bp, mats, storage)
#     # foreach(enumerate(bp)) do (ib, bk)
#     #     diagprodthree!(storage.diags,
#     #                    storage.M[ik],
#     #                    mats[ik][ib],
#     #                    adjoint(storage.M[bk]))
#     # end
#     for (ib, bk) in enumerate(bp)
#         diagprodthree!(storage.diags,
#                        storage.M[ik],
#                        mats[ik][ib],
#                        adjoint(storage.M[bk]))
#     end
# end


function objective!(AAA, C, storage)
    each_m_to_AAmat!(storage.M, AAA)
    res = 0.0
    # foreach(enumerate(C.bps)) do (ik, bp)
    #     fill!(storage.diags, 0.0)
    #     foreach(enumerate(bp)) do (ib, bk)
    #         diagprodthree!(storage.diags,
    #                        storage.M[ik],
    #                        C.mats[ik][ib],
    #                        adjoint(storage.M[bk]))
    #     end
    #     foreach(storage.diags) do nn
    #         res += 1-abs2(nn)
    #     end
    # end
    for (ik, bp) in enumerate(C.bps)
        fill!(storage.diags, 0.0)
        for (ib, bk) in enumerate(bp)
            diagprodthree!(storage.diags,
                           storage.M[ik],
                           C.mats[ik][ib],
                           adjoint(storage.M[bk]))
        end
        for nn in storage.diags
            res += 1-abs2(nn)
        end
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

function main()
    lnn = 4
    nk = 16
    nbs = 3
    nd = 2
    possible_lo = (4,5,6,7)

    los = rand(possible_lo, nk)
    bps = map(x->rand(eachindex(los), nbs), los)
    storage = init_storage(lnn, los)
    mats = init_mats(bps, los)
    C = (; bps, mats)
    AAA = init_AAA(los, nd)
    ∂AAA = Enzyme.make_zero(AAA)
    ∂storage = Enzyme.make_zero(storage)

    @time objective!(AAA, C, storage)
    s = objective!(AAA, C, storage)
    @time ∇objective!(∂AAA, AAA, C, storage, ∂storage)
    s, ∂AAA
end
