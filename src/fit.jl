
"""
	General purpose global optimisation function which is useful in Fingerprinting

"""
function global_optimisation(
	measured::NTuple{N, AbstractVector{<: Number}},
	model::Function,
	parameters::AbstractVector{<: Tuple{Real, Real}},
	iterations::Integer=20000
) where N

	# Define the loss function: overlap with the measured signal
	loss(p) = let loss = 0.0
		for (meas, sim) in zip(measured, model(p...))
			loss -= MRFingerprinting.overlap(meas, sim) / norm(sim)
		end
		return loss
	end

	# BlackBoxOptim
	result_blackbox = BlackBoxOptim.bboptimize(
		loss;
		SearchRange=parameters,
		MaxFuncEvals=iterations
	)
	result_blackbox = BlackBoxOptim.best_candidate(result_blackbox)

	# Simulated Annealing
	result_annealing = Optim.optimize(
		loss,
		[p[1] for p in parameters], # Lower
		[p[2] for p in parameters], # Upper
		mean.(parameters), # Initial parameters
		Optim.SAMIN(),
		Optim.Options(;iterations)
	)
	result_annealing = Optim.minimizer(result_annealing)
	return result_blackbox, result_annealing
end


@inline function overlap(a::AbstractVector{<: Number}, b::AbstractVector{<: Number})::Real
	abs(a' * b)
end
@inline function optimal_scale(a::AbstractVector{<: Number}, b::AbstractVector{<: Number})
	# Apply to a
	(a' * b) / (a' * a)
end
@inline function optimal_scale!(a::AbstractVector{<: Number}, b::AbstractVector{<: Number})
	a .*= (a' * b) / (a' * a)
end


function eval_loss(
	measured_signal::AbstractVector{<: Number},
	T1::Tuple{Real, Real},
	T2::Tuple{Real, Real},
	relB1::Real;
	# Parameters from here
	α::AbstractVector{<: Real},
	ϕ::AbstractVector{<: Real},
	TR::Union{Tuple{Real, Real}, AbstractVector{<: Real}},
	kmax::Integer,
	cycles::Integer,
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real
)
	# Not working

	ms = abs.(measured_signal) # This is bogus
	α = relB1 .* α

	R1 = 0.0002:0.0001:0.01
	R2 = 0.001:0.001:0.1
	result = Matrix{Float64}(undef, length(R2), length(R1))
	for (j, r1) = enumerate(R1)
		for (i, r2) = enumerate(R2)
			s2 = abs.(driven_equilibrium(α, ϕ, TR, kmax, cycles, G, τ, D, r1, r2)[1])
			result[i,j] = -(s2 ./ norm(s2))' * ms
		end
	end
	return result 
end

