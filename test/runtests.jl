using Test, GaussNewton


function rosenblock(m::Int, n::Int, x::AbstractArray{Float64}, r::AbstractArray{Float64}) 
	r[1] = 10 * (x[2] - x[1]^2)
	r[2] = 1 - x[1]
end

function brownbad(m::Int, n::Int, x::AbstractArray{Float64}, r::AbstractArray{Float64}) 
    r[1] = x[1] - 1000000
 	r[2] = x[2] - 2e-6
 	r[3] = x[1]*x[2] - 2
end


function beale(m::Int, n::Int, x::AbstractArray{Float64}, r::AbstractArray{Float64}) 
    r[1] = 1.5 - x[1]*(1-x[2])
 	r[2] = 2.25 - x[1]*(1-x[2]^2)
 	r[3] = 2.625 - x[1]*(1-x[2]^3)
end

@testset "GNRosenblock" begin
	gn = GN()
	x0 = [-1.2, 1.0]
    obj = rosenblock
	f0, info = optimize(obj, x0,  2, 2,gn)
	println(f0, " ", x0," ", info)
	@test f0 == 0 
	@test isapprox(1, x0[1], atol=1e-7)
    @test isapprox(1, x0[2], atol=1e-7)
end 


@testset "GNbrownbad" begin
gn = GN()
 	x0 = [1.0, 1.0]
 	f0, info = optimize(brownbad, x0,3, 2,gn)
 	println(f0, " ", x0," ", info)
    @test abs(f0) < eps(Float64)
    @test isapprox(1e6,x0[1],atol=1e-8)
    @test isapprox(2e-6,x0[2],atol=1e-8)
end

@testset "GNBeale" begin
    gn = GN()
    x0 = [1.0, 1.0]
 	f0, info = optimize(beale, x0, 3, 2, gn)
 	println(f0, " ", x0," ", info)
    @test abs(f0) < sqrt(eps(Float64))
    @test isapprox(3.0,x0[1],atol=gn.Reltol)
    @test isapprox(0.5,x0[2],atol=gn.Reltol)
end

 	
