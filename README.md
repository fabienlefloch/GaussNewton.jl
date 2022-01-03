# GaussNewton.jl
Klare and Miller Gauss-Newton Minimizer. Minimize sum of squares of residuals r using augmented Gauss-Newton step and Levenberg-Marquardt trust region.  Uses finite-difference derivatives and 1-D Jacobian updates.

This is a port of Fortran90 Netlib.org/misc/gn code originally from Kenneth Klare (kklare@gmail.com) and Guthrie Miller (guthriemiller@gmail.com).

## Usage

The simple syntax mirrors the Optim.jl package syntax

```julia
using GaussNewton
function rosenbrock(x)
	[1 - x[1], 100 * (x[2]-x[1]^2)]
end
x0 = zeros(2)
optimize(rosenbrock, x0)
```

A more optimized and powerful in place syntax is also possible, where the jacobian calculation may be passed as parameter:

```julia
using GaussNewton
function rosenbrock!(r,x)
	r[1] = 1 - x[1]
    r[2] = 100 * (x[2]-x[1]^2)
end
function rosenbrockDer!(J,x)
	J[1,1] = -1
    J[2,1] = -200*x[1]
    J[1,2] = 0
    J[2,2] = 100
end

x0 = zeros(2)
r = zeros(2) #output
optimize!(rosenbrock!, rosenbrockDer!, x0, r)
```
where the `rosenbrockDer!` is optional, and may be automatically computed through the autodiff parameter.

Additional optional parameters are:

 * stptol: step size for relative convergence test.
 * reltol, abstol: value relative/absolute convergence test.
 * derivstp in: the step for derivatives in autodiff = :single mode   Must be large enough to get some change in the function.    Must be small enough for some accuracy in the derivative.
 * limit  in: maximum number of all evaluations allowed, approximate.
 * tuning constants:
    -    ZLOW,ZHIGH change bounds of trust region (del) (.5,2).  Small changes in the path set by them greatly affect results.
    -    ZCP Cauchy step size (1).
    -    ZCPMIN minimum Cauchy step size (0.1).
    -    ZCPMAX maximum Cauchy step size (1000).
    -    MXBAD number bad steps with rank-1 updates (1).
* NewtStep1st=true start with Newton step. Use on linear problems.
* iscale=0, no scaling, iscale=1, variable scaling, iscale=2, fixed scaling based on D0, which must be allocated

The function returns the final sum of squares, and the minimum is stored in x0. In addition, it returns a detailed result structure from which success may be inferred through 

* `is_fatal(result::GNResult)` if a fatal error occured
* `has_converged(result::GNResult) = has_converged_abstol(result) || has_converged_reltol(result) || has_converged_stptol(result)`
* `has_converged_stptol(result::GNResult)`
* `has_converged_reltol(result::GNResult)`
* `has_converged_abstol(result::GNResult)`


## autodiff parameter

if the explicit jacobian calculation fcnDer! is not passed as a parameter (is nothing), then the parameter autodiff controls how the jacobian is calculated:

* autodiff = :forward uses the ForwarDiff package to compute the jacobian by automatic forward differentiation, 
* autodiff = :centered uses the FiniteDiff package to compute the jacobian
* autodiff = :single for single sided finite difference, as per the original code of Klare and Miller.

