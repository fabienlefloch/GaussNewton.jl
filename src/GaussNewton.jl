module GaussNewton

using LinearAlgebra, ForwardDiff, FiniteDiff, Printf

export optimize, optimize!, has_converged, is_fatal

struct GNTrace
    states::Vector{String}
end
GNTrace() = GNTrace(Vector{String}[])

function update(tr::GNTrace, iter::Int, f0::Real, x0::Vector, store_trace::Bool, show_trace::Bool, show_every::Int = 1)
    os = string("History ", iter, " ", f0, " ", x0)
    update(tr, os, store_trace, show_trace, show_every)
end

function update(tr::GNTrace, str::String, store_trace::Bool, show_trace::Bool, show_every::Int = 1)
    os = str
    if store_trace
        push!(tr.states, os)
    end
    if show_trace
        if show_every == 0
            show(os)
        end
    end
    return
end

struct GNResult{TR,TJ}
    ssr::Real #final sum of squares
    minimizer::TR
    best_jacobian::TJ
    info::Int
    iterations::Int
    reltol::Real
    abstol::Real
    stptol::Real
    tr::GNTrace
    f_calls::Int
    g_calls::Int
end

is_fatal(result::GNResult) = result.info >= 128
has_converged(result::GNResult) = has_converged_abstol(result) || has_converged_reltol(result) || has_converged_stptol(result)
has_converged_stptol(result::GNResult) = result.info & 0x1 == 0x1
has_converged_reltol(result::GNResult) = result.info & 0x2 == 0x2
has_converged_abstol(result::GNResult) = result.info & 0x8 == 0x8

function Base.show(io::IO, r::GNResult)
    @printf io "Results of Gauss-Newton Algorithm\n"
    @printf io " * Minimizer: [%s]\n" join(r.minimizer, ",")
    @printf io " * Sum of squares at Minimum: %f\n" r.ssr
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: %s\n" has_converged(r)
    @printf io " * |x - x'| < %.1e: %s\n" r.stptol has_converged_stptol(r)
    @printf io " * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" r.reltol has_converged_reltol(r)
    @printf io " * |f(x)| < %.1e: %s\n" r.abstol has_converged_abstol(r)
    @printf io " * Function Calls: %d\n" r.f_calls
    @printf io " * Gradient Calls: %d\n" r.g_calls
    @printf io " * Trace: %s\n" join(r.tr.states, "\n")
    return
end

#stptol in: step size for relative convergence test.
# ! reltol, abstol in: value relative/absolute convergence test.
# !    Setting accuracies too small wastes evaluations.
# !    Setting accuracies too large may not get the best solution.
# ! derivstp in: the step for derivatives.
# !    Must be large enough to get some change in the function.
# !    Must be small enough for some accuracy in the derivative.
# ! iprint in: degree of printout, if value is:
# !    0=none, 1=final f0 and x0, 2=iterations, 3=each x tried,
# !    4=all except Jacobian and Hessian, 5=all.
# ! ihist in: if > 0, write history file HistFile using unit ihist
# ! limit  in: maximum number of all evaluations allowed, approximate.
# ! tuning constants:
# !    ZLOW,ZHIGH change bounds of trust region (del) (.5,2).
# !    Small changes in the path set by them greatly affect results.
# !    ZCP Cauchy step size (1).
# !    ZCPMIN minimum Cauchy step size (0.1).
# !    ZCPMAX maximum Cauchy step size (1000).
# !    MXBAD number bad steps with rank-1 updates (1).
# ! NewtStep1st=.true. start with Newton step. Use on linear problems.
# ! mu0,del0,r0,H0,Jac0,iHOOKmax out: final values
# ! dfj/dxi=Jac0(j,i)
# ! iHOOKmax is max number of iterations in hook search
# ! iscale=0, no scaling
# ! iscale=1, variable scaling
# ! iscale=2, fixed scaling based on D0, which must be allocated
# !  and filled by user before call to GN

function optimize(fcn, x0::AbstractArray{T};
    show_trace::Bool = false,
    store_trace::Bool = false,
    NewtStep1st::Bool = false,
    iscale::Int = 0,
    limit::Int = 10000,
    abstol::T = 1e-11,
    derivstp::T = 1e-4,
    reltol::T = 3e-7,
    stptol::T = 1e-4,
    MXBAD::Int = 1,
    ZCONSEC::T = 1.005,
    ZHIGH::T = 2.0,
    ZLOW::T = 0.65,
    ZCP::T = 0.6,
    ZCPMAX::T = 1000.0,
    ZCPMIN::T = 0.18,
    D0 = zeros(T, length(x0)),
    autodiff = :single) where {T}
    fcn! = (out, x) -> copyto!(out, fcn(x))
    r = fcn(x0)
    return optimize!(fcn!, nothing, x0, zeros(eltype(r), length(r)),
        NewtStep1st = NewtStep1st,
        iscale = iscale,
        limit = limit,
        abstol = abstol,
        derivstp = derivstp,
        reltol = reltol,
        stptol = stptol,
        MXBAD = MXBAD,
        ZCONSEC = ZCONSEC,
        ZHIGH = ZHIGH,
        ZCP = ZCP,
        ZCPMAX = ZCPMAX,
        ZCPMIN = ZCPMIN,
        D0 = D0,
        autodiff = autodiff,
        show_trace = show_trace,
        store_trace = store_trace
    )

end
function optimize!(fcn!, x0::AbstractArray{T}, r::AbstractArray{TR};
    show_trace::Bool = false,
    store_trace::Bool = false,
    NewtStep1st::Bool = false,
    iscale::Int = 0,
    limit::Int = 10000,
    abstol::T = 1e-11,
    derivstp::T = 1e-4,
    reltol::T = 3e-7,
    stptol::T = 1e-4,
    MXBAD::Int = 1,
    ZCONSEC::T = 1.005,
    ZHIGH::T = 2.0,
    ZLOW::T = 0.65,
    ZCP::T = 0.6,
    ZCPMAX::T = 1000.0,
    ZCPMIN::T = 0.18,
    D0 = zeros(T, length(x0)),
    autodiff = :single) where {T,TR}

    return optimize!(fcn!, nothing, x0, r,
        NewtStep1st = NewtStep1st,
        iscale = iscale,
        limit = limit,
        abstol = abstol,
        derivstp = derivstp,
        reltol = reltol,
        stptol = stptol,
        MXBAD = MXBAD,
        ZCONSEC = ZCONSEC,
        ZHIGH = ZHIGH,
        ZCP = ZCP,
        ZCPMAX = ZCPMAX,
        ZCPMIN = ZCPMIN,
        D0 = D0,
        autodiff = autodiff,
        show_trace = show_trace,
        store_trace = store_trace
    )

end
function optimize!(fcn!, fcnDer!, x0::AbstractArray{T}, r::AbstractArray{TR};
    show_trace::Bool = false,
    store_trace::Bool = false,
    NewtStep1st::Bool = false,
    iscale::Int = 0,
    limit::Int = 10000,
    abstol::T = 1e-11,
    derivstp::T = 1e-4,
    reltol::T = 3e-7,
    stptol::T = 1e-4,
    MXBAD::Int = 1,
    ZCONSEC::T = 1.005,
    ZHIGH::T = 2.0,
    ZLOW::T = 0.65,
    ZCP::T = 0.6,
    ZCPMAX::T = 1000.0,
    ZCPMIN::T = 0.18,
    D0 = zeros(T, length(x0)),
    autodiff = :central) where {T,TR}
    #a port of fortran90 Netlib.org/misc/gn code
    # AUTHORS: Kenneth Klare (kklare@gmail.com)
    #    !      and Guthrie Miller (guthriemiller@gmail.com)
    #    ! OPEN SOURCE: may be freely used, with author achnowledgment,
    #    !    for any peaceful, beneficial purpose
    #    ! PURPOSE: Minimize sum of squares of residuals r using
    #    !    augmented Gauss-Newton step and Levenberg-Marquardt trust region.
    #    !    Uses finite-difference derivatives and 1-D Jacobian updates.
    #    ! KEYWORDS: Nonlinear Least squares, nonlinear minimization, robust,
    #    !           optimization, finite-difference derivatives,
    #    !           augmented Gauss-Newton, Levenberg-Marquardt.
    #    ! ARGUMENTS:
    #    ! fcn    in: A user subroutine to compute r(1:m) from x(1:n).
    #    !    It is invoked: call FCN(r,x), where
    #    !    x      in. n parameters to evaluate residuals.
    #    !    r      out. m residuals to be reduced. DO NOT SQUARE THEM.
    #    !    It must not change m, n, nor x.
    #    !    Scale the output residuals:
    #    !       r(i)=(fit(x,i)-y(i))/yerror(i), where y is the data.
    #    !    To reject the given x, set the r very large, like sqrt(largest).
    #    !    The starting point (x0) must valid and not be rejected.
    #    !    Declare FCN EXTERNAL so the compiler knows it.
    #    !    An iteration has function and derivative evaluations.
    #    !    Updates use function evaluations only.
    #    !    The program is designed to minimize evaluation count.
    #    ! m      in: number of equations or fitting residuals, the size of r0.
    #    ! n      in: number of variables, the size of x0.
    #    !    Program handles overdetermined, m>n, and underdetermined, m<n,
    #    !    as well as linear and nonlinear equations (m=n).
    #    ! info   out: status sum of bits (LT 128) or single error code (GE 128)
    #    !    Notation: pT q=transpose of p times q, ||q||=norm(q)=sqrt(qT q).
    #    ! CONVERGENCES
    #    !     1 step less than tolerance (stptol).
    #    !     2 rT r relative improvement less than tolerance (reltol).
    #    !     8 rT r sum of squares less than absolute tolerance (abstol).
    #    ! FATAL ERRORS
    #    !    128 input error.
    #    !    129 Jacobian is zero--cannot proceed with unchanging function.
    #    !    130 evaluation limit exceeded without termination.
    #    !    133 hook search error: could not find a suitable mu in 30 tries.
    #    ! x0     in/out n-vector: initial guess and best position found.
    #    ! f0     out scalar: final sum of squares. Think chi-squared.
    #    ! ACCESSIBLE in GN_MOD:
    #    ! r0     out m-vector: final residuals at x0
    #    ! iter   out scalar: actual number of iterations taken
    #    ! nfcn   out scalar: actual number of function evaluations used
    #    ! mu0    out scalar: the "best" L-M parameter used
    #    ! del0   out scalar: the "best" trust region size
    #    ! D0     out n-vector: scale factor, input for iscale=2
    #    ! H0     out n-by-n matrix: the unaugmented Hessian
    #    ! Jac0   out m-by-n array: the "best" Jacobian
    #    ! limitATIONS:
    #    !    Strict descent limits use to reproducible data.
    #    !    x0, r0, and others must be 1-D vectors, not arrays.
    #    !    Large m and n will require large storage and round-off errors.
    #    !    Your residuals should not exceed sqrt(HUGE/m) to avoid overflows.
    #    ! RECOMMENDATIONS:
    #    !    Use double-precision calculation, REAL*8, if possible.
    #    !    Single precision can do 3 to 4 digits in x, at best.
    #    !    Give a good guess or you may find a secondary minimum.
    #    !    Scale parameters and residuals, so chi-square and relative
    #    !    errors are meaningful.
    #    ! CREDITS:
    #    !     * Dennis and Schnabel, "Numerical Methods for Unconstrained
    #    !     Optimization and Nonlinear Equations" (1983),
    #    !     for many ideas and the hook calculation, i.e., mu from phi.
    #    !     * IMSL ZXSSQ for the Jacobian update attributed in
    #    !     Fletcher, "Practical Methods of Optimization" (1980)
    #    !     to Barnes (1965) and Broyden (1965).
    #    !     * x(+) = x-(JacT Jac+mu I)**-1 JacT r augmented Gauss-Newton.
    #    !     * See this for allocations and vector operators in F90.
    #    !     <http://www.stanford.edu/class/me200c/tutorial_90/07_arrays.html>
    #    !     * See "Fortran90 for Fortran77 Programmers" by Clive Page
    #    !     <http://www.star.le.ac.uk/~cgp/f90course/f90.html> 11/26/2001
    #    ! DYNAMICALLY ALLOCATED:
    #    ! Jac  scratch, m-by-n array: Jacobian of derivatives w.r.t. variables.
    #    ! H    scratch, n-by-n array: JacT Jac Hessian without 2nd order derivs.
    #    ! L    scratch, n-by-n array: lower-triangle factor, L LT = H+mu*I.
    #    ! r    scratch, m vector of residuals. ChiSq=Sum(rT r).
    #    ! D,g,sN,s,x   scratch, n vectors.
    #    !    They are scale factor, gradient, Newton step, step tried, position.
    #    !    You must scale by scaling factors D.
    #    !    Use dfj/dxi=Jac(j,i)/D(i) H(i,j)/D(i)/D(j) g(j)/D(j) sN(j)*D(j).
    local hook = false
    local take = false
    local iHOOK = 0
    local nbad = 0
    local nconsec = 0
    local info = 0
    local del = zero(T)
    local f::T
    local f0 = zero(T)
    local sfp::T
    local gnorm::T
    local mu::T
    local mulow::T
    local muhigh::T
    local phi::T
    local phip::T
    local rr::T
    local scal::T
    local scal0::T
    local snewt::T
    local snorm::T
    local temp::T

    m = length(r)
    n = length(x0)

    local tracing = show_trace || store_trace
    local tr = GNTrace()
    r0 = zeros(T, m)
    Jac0 = zeros(T, (m, n))
    H0 = zeros(T, (n, n))
    local Jac = zeros(T, (m, n))
    local H = zeros(T, (n, n))
    local L = zeros(T, (n, n))
    local D = ones(T, n)
    local g = zeros(T, n)
    local s = zeros(T, n)
    local sN = zeros(T, n)
    local epsn12 = sqrt(eps(T) * n)
    local x = copy(x0)
    local newFcnDer! = fcnDer!
    if iscale == 2
        if length(D0) != n
            throw(error("the size of D0 must match n"))
        elseif norm(D0) == 0
            throw(error("the norm of D0 can not be zero"))
        else
            D[1:end] = D0
        end
    end

    if typeof(fcnDer!) == Nothing
        if autodiff == :central
            central_cache = FiniteDiff.JacobianCache(similar(x), similar(r), similar(r))
            newFcnDer! = (J::Matrix, xp::Vector) -> FiniteDiff.finite_difference_jacobian!(J, fcn!, xp, central_cache)
        elseif autodiff == :forward
            jac_cfg = ForwardDiff.JacobianConfig(fcn!, r, x, ForwardDiff.Chunk(x))
            ForwardDiff.checktag(jac_cfg, fcn!, x)
            newFcnDer! = (J::Matrix, xp::Vector) -> ForwardDiff.jacobian!(J, fcn!, deepcopy(r), xp, jac_cfg, Val{false}())
        elseif autodiff == :single
            newFcnDer! = nothing
        else
            throw(DomainError(autodiff, "Invalid automatic differentiation method."))
        end
    end

    nfcn = 0
    nfcnDer = 0
    nrank1 = n
    hook = false
    take = false
    nconsec = 0
    nconsecMax = 0
    iHOOKmax = 0

    hnorm = zero(T)
    js = zeros(T, m)
    itersave = 0
    f_calls = 0
    g_calls = 0
    for iter = 1:limit
        itersave = iter
        info = 0
        nfcn += 1
        f_calls += 1
        fcn!(r, x)
        f = norm(r)^2
        if iter > 1
            take = f < f0
            if take
                nconsec += 1
                nconsec = max(nconsecMax, nconsec)
            else
                nconsec = 0
            end
        end
        if iter == 1
            take = true
        elseif iter == 2 && NewtStep1st && !take
            # ! CAUCHY STEP is distance to Cauchy point in Newton direction.
            # ! Here if Newton step didn't work.
            hook = false
            del = cauchyDist(n, L, g, gnorm, ZCP = ZCP, ZCPMIN = ZCPMIN, ZCPMAX = ZCPMAX)
        else
            #       ! UPDATE TRUST REGION.
            # ! Strict descent requires f<f0.
            # ! Parabolic interpolation/extrapolation using f0, f and slope at x0.
            # ! Update del to be distance to minimum, new del is delFac times old.
            # lastdel = del
            if abs(f - f0) <= max(reltol * max(f, f0), abstol)
                info = 2
            end
            # ! sfp is the step length s times the slope fp of f in the direction s.
            # ! it would be the change in f for negligible curvature.
            # ! sfp is negative because the step is in a decreasing direction.
            sfp = 2 * dot(g, s)
            rr = (f - f0) / sfp
            # ! rr is bounded above by 1, when f < f0 and f behaves linearly,
            # ! and unbounded below when f > f0
            delFac = 1 / (2 * max(1 - rr, eps(T)))
            del = snorm * min(max(delFac, ZLOW), ZHIGH)
            # ! we also increase trust radius based on number of consecutive takes.
            # ! Test problem #36 illustrates the need for this.
            del = del * ZCONSEC^nconsec
        end
        if take
            # ! SAVE THE BEST.
            #    if(iprint.ge.4) write(*,*) 'saving best'\
            tracing && update(tr, "saving best", store_trace, show_trace)
            nbad = 0
            x0[1:end] = x
            f0 = f
            # mu0 = mu
            # del0 = del
            D0[1:end] = D
            H0[1:end, 1:end] = H
            Jac0[1:end, 1:end] = Jac

            # if(ihist.gt.0) write(ihist,*) iter,f0,x0
            tracing && update(tr, iter, f0, x0, store_trace, show_trace)
        elseif iter != 2 || !NewtStep1st
            # ! Overshot the mark. Try, then get a new Jacobian.
            hook = true
            if nrank1 > 0
                nbad += 1
            end
            #if(iprint.ge.4) write(*,*)'Overshot nrank1,nbad',nrank1,nbad
            tracing && update(tr, string("Overshot ", nrank1, " ", nbad), store_trace, show_trace)
        end
        # ! Some convergence tests.
        if iter > 1
            if snorm <= stptol
                info = info + 1
            end
        end
        if f < abstol
            info = info + 8
        end
        done = (info != 0 && nrank1 == 0) || info >= 8
        if !done && nfcn >= limit
            info = 130
        end
        if iter > 1 && (done || info >= 128)
            break
        end
        # ! ---Jacobian dfi/dxj, gradient g=JacT r, Hessian H=JacT Jac+2nd order
        # ! Jacobian update when not stale nor final, else a full Jacobian.
        if nbad <= MXBAD && info == 0 && nrank1 < n
            if !take
                @goto L40
            end
            # ! Rank-1 update to Jacobian. Jac(new) = Jac+((r-r0-Jac s) sT)/(sT s).
            #     if(iprint.ge.4) write(*,*)
            # 1     'Rank=1 Jacobian update: nrank1,nbad,snorm',nrank1,nbad,snorm
            tracing && update(tr, string("Rank=1 Jacobian update: ", nrank1, " ", nbad, " ", snorm), store_trace, show_trace)
            nrank1 = nrank1 + 1
            js = Jac * s
            #Jac = Jac +matmul(reshape(r-r0-matmul(Jac,s),(m,1)),               reshape(s/snorm**2,(1,n))) //!outer product
            for u = 1:m
                for v = 1:n
                    Jac[u, v] += (r[u] - r0[u] - js[u]) * (s[v] / (snorm * snorm))
                end
            end
            if take
                r0[1:end] = r
            end
        else

            # ! Full Jacobian.
            # ! Step away from zero to avoid crossing it.
            #          if(iprint.ge.4) write(*,*)'Full Jacobian: nrank1,nbad,take'
            #      1                                            ,nrank1,nbad,take
            tracing && update(tr, string("Full Jacobian: ", nrank1, " ", nbad, " ", take), store_trace, show_trace)
            nrank1 = 0
            nbad = 0
            if take
                r0[1:end] = r
            end
            nfcn += n
            g_calls += 1
            if iscale == 1
                # ! variable scale
                scal0 = dot(s, D)^2
                @. D = max((abs(x0) + D) / 2, stptol)

                # if(iprint.ge.4) write(*,*) 'variable scale D',D
                tracing && update(tr, string("variable scale ", D), store_trace, show_trace)
                scal = dot(s, D)^2
                if scal > zero(T)
                    del = del * sqrt(scal0 / scal)
                end
            end
            if typeof(newFcnDer!) == Nothing
                computeGNJacobian(fcn!, x0, D, r0, r, Jac, derivstp = derivstp)
            else
                newFcnDer!(Jac, x0)
                for j = 1:n
                    dj = D[j]
                    Jac[1:end, j] .*= dj
                end
            end

        end

        # ! Gradient and Hessian.
        g = Jac' * r0
        H = Jac' * Jac
        #  if(iprint.ge.5) then
        #     do i=1,m
        #        write(*,*) 'J',Jac(i,:)
        #     enddo
        #     do j=1,n
        #        write(*,*) 'H',H(j,:)
        #     enddo
        #  endif
        gnorm = norm(g)
        #       if(iprint.ge.4) write(*,*) 'gnorm,g',gnorm,g
        tracing && update(tr, string(gnorm, " ", g), store_trace, show_trace)
        # ! L1 norm (max of row sums of abs) of H(i,j).
        # ! H=JacT Jac symmetric.
        hnorm = zero(T)
        for j = 1:n
            lsum = sum(x -> abs(x), H[1:end, j])
            hnorm = max(hnorm, lsum)
        end
        # ! Get a small number for further augmentation, check underflow.
        hnorm = hnorm * epsn12
        if hnorm == 0
            info = 129
            break #MainLoop
        end
        # ! Find bad rows and ignore them.
        for j = 1:n
            if H[j, j] <= 0
                H[j, j] = one(T)
                H[j, :] .= zero(T)
                H[:, j] .= zero(T)
            end
        end

        # ! --- GET NEWTON STEP, H sN = -g ---
        mu = zero(T)
        # ! solve (H + mu I)s = g for step s, possibly augmenting diagonal of H
        GN_CHOL(n, mu, H, g, L, sN, hnorm)
        snewt = norm(sN)
        if snewt <= del
            hook = false
        end
        # if(iprint.ge.4) write(*,*) 'snewt,mu,sN',snewt,mu,sN
        tracing && update(tr, string(snewt, " ", mu, " ", sN), store_trace, show_trace)
        if iter == 1
            if NewtStep1st
                # ! Try Newton step first.
                del = snewt
                s[1:end] = sN
                snorm = snewt
                # if(iprint.ge.4) write(*,*)'Taking Newton step first'
                tracing && update(tr, "Taking Newton step first", store_trace, show_trace)
                @goto L100
            else
                # ! more conservative approach, try Cauchy step first
                hook = false
                del = cauchyDist(n, L, g, gnorm, ZCP = ZCP, ZCPMIN = ZCPMIN, ZCPMAX = ZCPMAX)
            end
        end
        @label L40  # continue ! --- CHOOSE NEWTON OR HOOK STEPS -------------------
        if !hook
            # ! Allow Newton step to be up to the current trust radius del
            temp = 1
            if snewt > del
                temp = del / snewt
            end
            s = sN .* temp
            snorm = snewt * temp
            #     if(iprint.ge.4) write(*,*) 'Step in Newton direction:'
            # 1    ,' snewt,del,snorm',snewt,del,snorm
            tracing && update(tr, string("Step in Newton direction: ", snewt, " ", del, " ", snorm), store_trace, show_trace)
        else
            # ! --- Hook search ---
            # ! Find step of length del by finding mu
            # ! that gives snorm = ||s|| = ||(H + mu I)**-1 g|| = del.
            # ! Because del = ||(H + mu I)**-1 g|| and H is positive,
            # ! del is less than ||g||/mu.
            muhigh = gnorm / del
            mulow = zero(T)
            # ! mulow <= mu <= muhigh
            # muStart = mu
            # ! REPEAT UNTIL abs(del-||s||)<.05 or mulow>muhigh.
            #   HookLoop: do iHOOK=1,30
            for iHOOK = 1:30
                GN_CHOL(n, mu, H, g, L, s, hnorm)
                mulow = max(mulow, mu)
                snorm = norm(s)
                if abs(snorm - del) <= 0.05 * del || mulow >= muhigh
                    break
                end
                # 1       exit HookLoop
                phi = snorm - del
                # ! phi<0 indicates mu is too large, use this mu to update muhigh
                if phi < 0
                    muhigh = min(mu, muhigh)
                end
                if phi > 0
                    mulow = max(mu, mulow)
                end
                # ! Want sT (H + mu I)**-1 s = ||L**-1 s||**2.
                # ! Start with x = L**-1 s.
                GN_LSOLVE(n, L, x, s)

                phip = zero(T)
                if snorm > zero(T)
                    sx = norm(x)^2
                    phip = -sx / snorm
                end
                #    ! As mu increases the step size decreases, so phip must be negative.
                if phip < zero(T)
                    mu = mu - (snorm / del) * (phi / phip)
                end
                mu = max(min(mu, muhigh), mulow)
            end
            #! End of REPEAT.

            #     if(iprint.ge.4.or.iHOOK.ge.30) then
            #        write(*,*) 'iHOOK,iHOOKmax,muStart,mu,lastdel,del,snorm'
            # 1                 ,iHOOK,iHOOKmax,muStart,mu,lastdel,del,snorm
            tracing && update(tr, string(iHOOK, " ", iHOOKmax, " ", mu, " ", del, " ", snorm), store_trace, show_trace)
            if iHOOK >= 30
                info = 133
                if iHOOK > 30
                    break # MainLoop
                end
            end
            iHOOKmax = max(iHOOKmax, iHOOK)
        end

        @label L100 #            continue ! TAKE THE STEP.
        # if(iprint.ge.4) write(*,*) 'Taking step s',s
        tracing && update(tr, string("Taking step ", s), store_trace, show_trace)
        @. x = x0 + s * D
    end
    # abs(f - f0) <= max(reltol * max(f, f0), abstol)
    return f0, GNResult(f0, x0, Jac0, info, itersave, reltol, abstol, stptol, tr, f_calls, g_calls)
end





function computeGNJacobian(fcn, x0::AbstractArray{T}, D::AbstractArray{T}, r0::AbstractArray{T}, r::AbstractArray{T}, Jac::AbstractMatrix{T}; derivstp::T = 1e-4) where {T}
    #start jacobian
    for j = 1:length(x0)
        hold = x0[j]
        step = copysign(derivstp, hold)
        x0[j] += step * D[j]
        fcn(r, x0) #jacevaluation
        Jac[:, j] = (r - r0) / step
        x0[j] = hold
    end
    #end jacobian => could be analytical
end



# !-----------------------------------------------------------------------
# !+Cauchy distance is distance along gradient to minimum.
# ! Cauchy step is Cauchy distance in the Newton direction.
# ! Cauchy distance = -||g||**2/(gT H g) ||g||, gT H g = ||LT g||**2.
# ! Use ZCP * Cauchy distance.
# ! ||v|| = sqrt(vT v) is L2 or Euclidean norm of vector v.
function cauchyDist(n::Int, L::AbstractArray{T}, g::AbstractArray{T}, gnorm::T; ZCP::T, ZCPMIN::T, ZCPMAX::T)::T where {T}
    # use GN_MOD, only: ZCP,ZCPMIN,ZCPMAX,iprint
    # implicit none
    # intrinsic max,min,sum
    # real*8 L(n,n),g(n),gnorm,del,temp
    if gnorm == zero(T)
        return ZCPMIN
    end
    temp = zero(T)
    # ! calculate temp = gT H g/||g||**2 = ||LT g||**2/||g||**2
    for j = 1:n
        sum = dot(L[j:end, j], g[j:end])
        temp += (sum / gnorm)^2
    end
    temp = ZCP * gnorm / temp
    del = max(ZCPMIN, min(temp, n * ZCPMAX))
    #  if(iprint.ge.4) write(*,*)
    # 1 'Cauchy step,del,stpmax',temp,del,n*ZCPMAX
    return del
end #function GN_CauchyDist

# !-----------------------------------------------------------------------
# !+Cholesky decomposition and solution, (H + (mu+add)*I) s = -g.
# ! Decomposition to lower triangle L without addition if it works.
# ! Break as soon as OK.
function GN_CHOL(n::Int, mu::T, H::AbstractMatrix{T}, g::AbstractArray{T}, L::AbstractMatrix{T}, s::AbstractArray{T}, add::T) where {T}
    # implicit none
    # intrinsic sqrt,min,sum
    # integer n,i,iadd,j
    # real*8 add,mu,tmp,H(n,n),g(n),L(n,n),s(n)
    #loop1:
    isCycle = false
    for iadd = 1:n
        for j = 1:n
            for i = j:n
                sum = dot(L[i, 1:j-1], L[j, 1:j-1])
                L[i, j] = H[j, i] - sum
            end
            tmp = L[j, j] + mu
            if tmp <= 0
                mu = mu + add
                isCycle = true # loop1
                break
            end
            tmp = sqrt(tmp)
            L[j, j] = tmp
            L[j+1:end, j] ./= tmp
        end
        if !isCycle
            break
        else
            isCycle = false
        end
    end
    # ! forward row reduction and backward substitution
    for i = 1:n
        sum = dot(L[i, 1:i-1], s[1:i-1])
        s[i] = (-g[i] - sum) / L[i, i]
    end
    for i = n:-1:1
        sum = dot(L[i+1:end, i], s[i+1:end])
        s[i] = (s[i] - sum) / L[i, i]
    end
end

# !-----------------------------------------------------------------------
# !+solve L x = y. x and y may be the same vector, L lower triangle.
function GN_LSOLVE(n::Int, L::AbstractMatrix{T}, x::AbstractArray{T}, y::AbstractArray{T}) where {T}
    # implicit none
    # intrinsic sum
    # integer n,i
    # real*8 L(n,n),x(n),y(n)
    for i = 1:n
        sum = dot(L[i, 1:i-1], x[1:i-1])
        x[i] = (y[i] - sum) / L[i, i]
    end
end #subroutine GN_LSOLVE

end # module
