
module MRFingerprinting

	using LinearAlgebra
	using LinearOperators
	using LoopVectorization
	using LoopVectorizationTools
	using FFTW
	FFTW.set_num_threads(Threads.nthreads())
	import FunctionWrappers: FunctionWrapper
	import MRIRecon
	using Statistics
	import Optim
	import BlackBoxOptim

	include("overlap.jl")
	include("fit.jl")
	include("dictionary.jl")
	include("matching.jl")
	include("low_rank.jl")

end

