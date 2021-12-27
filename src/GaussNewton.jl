module GaussNewton

using LinearAlgebra

export GN, optimize

struct GN{T}
    NewtStep1st::Bool
    Iscale::Int
    Limit::Int
    Abstol::T
    Derivstp::T
    Reltol::T
    Stptol::T
    MXBAD::Int
    ZCONSEC::T
    ZHIGH::T
    ZLOW::T
    ZCP::T
    ZCPMAX::T
    ZCPMIN::T

    function GN()
        return new{Float64}(false, 0, 10000, 1e-11, 1e-4, 3e-7, 1e-4, 1, 1.005, 2.0, 0.65, 0.6, 1000.0, 0.18)
    end
end


#Stptol in: step size for relative convergence test.
# ! Reltol, Abstol in: value relative/absolute convergence test.
# !    Setting accuracies too small wastes evaluations.
# !    Setting accuracies too large may not get the best solution.
# ! Derivstp in: the step for derivatives.
# !    Must be large enough to get some change in the function.
# !    Must be small enough for some accuracy in the derivative.
# ! iprint in: degree of printout, if value is:
# !    0=none, 1=final f0 and x0, 2=iterations, 3=each x tried,
# !    4=all except Jacobian and Hessian, 5=all.
# ! ihist in: if > 0, write history file HistFile using unit ihist
# ! Limit  in: maximum number of all evaluations allowed, approximate.
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
# ! Iscale=0, no scaling
# ! Iscale=1, variable scaling
# ! Iscale=2, fixed scaling based on D0, which must be allocated
# !  and filled by user before call to GN

function optimize(fcn,  x0::AbstractArray{T}, m::Int, n::Int, gn::GN{T}) where {T}
    return optimize(fcn, nothing, x0, m, n, gn)
end

function optimize(fcn, fcnDer, x0::AbstractArray{T}, m::Int, n::Int, gn::GN{T}; D0 =  zeros(T, n)) where {T}
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
    #    !    It is invoked: call FCN(m,n,x,r), where
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
    #    !     1 step less than tolerance (Stptol).
    #    !     2 rT r relative improvement less than tolerance (Reltol).
    #    !     8 rT r sum of squares less than absolute tolerance (Abstol).
    #    ! FATAL ERRORS
    #    !    128 input error.
    #    !    129 Jacobian is zero--cannot proceed with unchanging function.
    #    !    130 evaluation Limit exceeded without termination.
    #    !    133 hook search error: could not find a suitable mu in 30 tries.
    #    ! x0     in/out n-vector: initial guess and best position found.
    #    ! f0     out scalar: final sum of squares. Think chi-squared.
    #    ! ACCESSIBLE in GN_MOD:
    #    ! r0     out m-vector: final residuals at x0
    #    ! iter   out scalar: actual number of iterations taken
    #    ! nfcn   out scalar: actual number of function evaluations used
    #    ! mu0    out scalar: the "best" L-M parameter used
    #    ! del0   out scalar: the "best" trust region size
    #    ! D0     out n-vector: scale factor, input for Iscale=2
    #    ! H0     out n-by-n matrix: the unaugmented Hessian
    #    ! Jac0   out m-by-n array: the "best" Jacobian
    #    ! LimitATIONS:
    #    !    Strict descent Limits use to reproducible data.
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
    local  nbad = 0
    local  nconsec = 0
    local info = 0
    local del = zero(T)
    local  epsn12::T
    local  f::T
    local f0 = zero(T)
    local  sfp::T
    local gnorm::T
    local  mu::T
    local  mulow::T
    local  muhigh::T
    local phi::T
    local  phip::T
    local  rr::T
    local  scal::T
    local  scal0::T
    local  snewt::T
    local  snorm::T
    local  temp::T

    local Jac::AbstractMatrix{T}
    local H::AbstractMatrix{T}
    local L::AbstractMatrix{T}
    local D::AbstractArray{T}
    local  g::AbstractArray{T}
    local  r::AbstractArray{T}
    local  s::AbstractArray{T}
    local  sN::AbstractArray{T}
    local  x::AbstractArray{T}

    r0 = zeros(T, m)
    
    Jac0 = zeros(T, (m, n))
    H0 = zeros(T, (n, n))
    Jac = zeros(T, (m, n))
    H = zeros(T, (n, n))
    L = zeros(T, (n, n))
    r = zeros(T, m)
    D = ones(T, n)
    g = zeros(T, n)
    s = zeros(T, n)
    sN = zeros(T, n)
    epsn12 = sqrt(eps(T) * n)
    x = copy(x0)

    if gn.Iscale == 2
        if length(D0) != n 
            throw(error("the size of D0 must match n"))
        elseif norm(D0) == 0
            throw(error("the norm of D0 can not be zero"))
        else
            D[1:end] = D0
        end
    end
    nfcn = 0
    nrank1 = n
    hook = false
    take = false
    nconsec = 0
    nconsecMax = 0
    iHOOKmax = 0

    hnorm = zero(T)
    js = zeros(T, m)
    for iter = 1:gn.Limit
        info = 0
        nfcn += 1
        fcn(m, n, x, r)
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
        elseif iter == 2 && gn.NewtStep1st && !take
            # ! CAUCHY STEP is distance to Cauchy point in Newton direction.
            # ! Here if Newton step didn't work.
            hook = false
            del = cauchyDist(n, L, g, gnorm, ZCP = gn.ZCP, ZCPMIN = gn.ZCPMIN, ZCPMAX = gn.ZCPMAX)
        else
            #       ! UPDATE TRUST REGION.
            # ! Strict descent requires f<f0.
            # ! Parabolic interpolation/extrapolation using f0, f and slope at x0.
            # ! Update del to be distance to minimum, new del is delFac times old.
            # lastdel = del
            if abs(f - f0) <= max(gn.Reltol * max(f, f0), gn.Abstol)
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
            del = snorm * min(max(delFac, gn.ZLOW), gn.ZHIGH)
            # ! we also increase trust radius based on number of consecutive takes.
            # ! Test problem #36 illustrates the need for this.
            del = del * gn.ZCONSEC^nconsec
        end
        if take
            # ! SAVE THE BEST.
            #    if(iprint.ge.4) write(*,*) 'saving best'
            nbad = 0
            x0[1:end] = x
            f0 = f
            # mu0 = mu
            # del0 = del
            D0[1:end] = D
            H0[1:end, 1:end] = H
            Jac0[1:end, 1:end] = Jac

            # if(ihist.gt.0) write(ihist,*) iter,f0,x0
        elseif iter != 2 || !gn.NewtStep1st
            # ! Overshot the mark. Try, then get a new Jacobian.
            hook = true
            if nrank1 > 0
                nbad += 1
            end
            #if(iprint.ge.4) write(*,*)'Overshot nrank1,nbad',nrank1,nbad
        end
        # ! Some convergence tests.
        if iter > 1
            if snorm <= gn.Stptol
                info = info + 1
            end
        end
        if f < gn.Abstol
            info = info + 8
        end
        done = (info != 0 && nrank1 == 0) || info >= 8
        if !done && nfcn >= gn.Limit
            info = 130
        end
        if iter > 1 && (done || info >= 128)
            break
        end
        # ! ---Jacobian dfi/dxj, gradient g=JacT r, Hessian H=JacT Jac+2nd order
        # ! Jacobian update when not stale nor final, else a full Jacobian.
        if nbad <= gn.MXBAD && info == 0 && nrank1 < n
            if !take
                @goto L40
            end
            # ! Rank-1 update to Jacobian. Jac(new) = Jac+((r-r0-Jac s) sT)/(sT s).
            #     if(iprint.ge.4) write(*,*)
            # 1     'Rank=1 Jacobian update: nrank1,nbad,snorm',nrank1,nbad,snorm
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
            nrank1 = 0
            nbad = 0
            if take
                r0[1:end] = r
            end
            nfcn = nfcn + n
            if gn.Iscale == 1
                # ! variable scale
                scal0 = dot(s, D)^2
                @. D = max((abs(x0) + D) / 2, gn.Stptol)

                # if(iprint.ge.4) write(*,*) 'variable scale D',D
                scal = dot(s, D)^2
                if scal > 0
                    del = del * sqrt(scal0 / scal)
                end
            end
            if fcnDer == nothing
                computeGNJacobian(fcn, m, n, x0, D, r0, r, Jac, derivstp = gn.Derivstp)
            else
                fcnDer(m, n, x0, Jac)
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

        if iter == 1
            if gn.NewtStep1st
                # ! Try Newton step first.
                del = snewt
                s[1:end] = sN
                snorm = snewt
                # if(iprint.ge.4) write(*,*)'Taking Newton step first'
                @goto L100
            else
                # ! more conservative approach, try Cauchy step first
                hook = false
                del = cauchyDist(n, L, g, gnorm, ZCP = gn.ZCP, ZCPMIN = gn.ZCPMIN, ZCPMAX = gn.ZCPMAX)
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
                if snorm > 0
                    sx = norm(x)^2
                    phip = -sx / snorm
                end
                #    ! As mu increases the step size decreases, so phip must be negative.
                if phip < 0
                    mu = mu - (snorm / del) * (phi / phip)
                end
                mu = max(min(mu, muhigh), mulow)
            end
            #! End of REPEAT.

            #     if(iprint.ge.4.or.iHOOK.ge.30) then
            #        write(*,*) 'iHOOK,iHOOKmax,muStart,mu,lastdel,del,snorm'
            # 1                 ,iHOOK,iHOOKmax,muStart,mu,lastdel,del,snorm
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
        @. x = x0 + s * D



    end

    return f0, info
end





function computeGNJacobian(fcn, m::Int, n::Int, x0::AbstractArray{T}, D::AbstractArray{T}, r0::AbstractArray{T}, r::AbstractArray{T}, Jac::AbstractMatrix{T}; derivstp::T = 1e-4) where {T}
    #start jacobian
    for j = 1:n
        hold = x0[j]
        step = copysign(derivstp, hold)
        x0[j] += step * D[j]
        fcn(m, n, x0, r) #jacevaluation
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
        return gn.ZCPMIN
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
function GN_CHOL(n::Int, mu::T, H::AbstractMatrix{T}, g::AbstractArray{T}, L::AbstractMatrix{T}, s::AbstractArray{T},add::T) where {T}
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
