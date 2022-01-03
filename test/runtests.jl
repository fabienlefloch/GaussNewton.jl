using Test, GaussNewton


function rosenblock(r::AbstractArray{Float64}, x::AbstractArray{Float64})
    r[1] = 10 * (x[2] - x[1]^2)
    r[2] = 1 - x[1]
end

function brownbad(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
    r[1] = x[1] - 1000000
    r[2] = x[2] - 2e-6
    r[3] = x[1] * x[2] - 2
end


function beale(r::AbstractArray{Float64}, x::AbstractArray{Float64})
    r[1] = 1.5 - x[1] * (1 - x[2])
    r[2] = 2.25 - x[1] * (1 - x[2]^2)
    r[3] = 2.625 - x[1] * (1 - x[2]^3)
end

@testset "GNRosenblock" begin
    x0 = [-1.2, 1.0]
    obj = rosenblock
    f0, info = optimize!(obj, x0, zeros(2))
    println(f0, " ", x0, " ", info)
    @test f0 == 0
    @test isapprox(1, x0[1], atol = 1e-7)
    @test isapprox(1, x0[2], atol = 1e-7)
    @test has_converged(info) && !is_fatal(info)
end


@testset "GNbrownbad" begin
    x0 = [1.0, 1.0]
    f0, info = optimize!(brownbad, x0, zeros(3))
    println(f0, " ", x0, " ", info)
    @test abs(f0) < eps(Float64)
    @test isapprox(1e6, x0[1], atol = 1e-8)
    @test isapprox(2e-6, x0[2], atol = 1e-8)
    @test has_converged(info) && !is_fatal(info)

    x0 = [1.0, 1.0]
    f0, info = optimize!(brownbad, x0, zeros(3), autodiff = :forward)
    println(f0, " ", x0, " ", info)
    @test abs(f0) < eps(Float64)
    @test isapprox(1e6, x0[1], atol = 1e-8)
    @test isapprox(2e-6, x0[2], atol = 1e-8)
    @test has_converged(info) && !is_fatal(info)

    x0 = [1.0, 1.0]
    f0, info = optimize!(brownbad, x0, zeros(3), autodiff = :central)
    println(f0, " ", x0, " ", info)
    @test abs(f0) < eps(Float64)
    @test isapprox(1e6, x0[1], atol = 1e-8)
    @test isapprox(2e-6, x0[2], atol = 1e-8)
    @test has_converged(info) && !is_fatal(info)

end

@testset "GNBeale" begin
    x0 = [1.0, 1.0]
    f0, info = optimize!(beale, x0, zeros(3))
    println(f0, " ", x0, " ", info)
    @test abs(f0) < sqrt(eps(Float64))
    @test isapprox(3.0, x0[1], atol = 1e-7)
    @test isapprox(0.5, x0[2], atol = 1e-7)
    @test has_converged(info) && !is_fatal(info)
end


