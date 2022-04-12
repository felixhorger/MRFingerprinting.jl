
module MRFingerprinting

	using LinearAlgebra
	using LoopVectorization
	using Statistics
	using FFTW
	using IterativeSolvers
	using LinearMaps
	import Optim
	import BlackBoxOptim

	include("fit.jl")
	include("dictionary.jl")
	include("overlap.jl")
	include("matching.jl")
	include("low_rank_recon.jl")

end

