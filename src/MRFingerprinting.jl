
module MRFingerprinting

	using LinearAlgebra
	using LoopVectorization
	using Statistics
	import Optim
	import BlackBoxOptim
	import MRIEPG

	include("fit.jl")
	include("dictionary.jl")

end

