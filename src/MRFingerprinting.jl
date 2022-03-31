
module MRFingerprinting

	using LinearAlgebra
	using LoopVectorization
	using Statistics
	import Optim
	import BlackBoxOptim

	include("fit.jl")
	include("dictionary.jl")

end

