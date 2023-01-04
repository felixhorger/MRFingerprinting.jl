
module MRFingerprinting

	using Base.Cartesian
	using LinearAlgebra
	using LinearOperators
	using LoopVectorization
	using Statistics
	using IterativeSolvers
	using FFTW
	FFTW.set_num_threads(Threads.nthreads())
	import Optim
	import BlackBoxOptim
	import FunctionWrappers: FunctionWrapper
	import MRIRecon

	include("overlap.jl")
	include("fit.jl")
	include("dictionary.jl")
	include("matching.jl")
	include("low_rank.jl")
	include("turbo_copy.jl")

end

