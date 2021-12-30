
module MRFingerprinting

	import Optim
	import BlackBoxOptim
	import MRIEPG
	using LinearAlgebra
	using Statistics

	macro global_optimisation_run(search_range, lower, upper, initial)
		esc(quote
			# Put boundaries in Relaxometry
			R1 = 1 ./ reverse(T1)
			R2 = 1 ./ reverse(T2)

			result_blackbox = BlackBoxOptim.bboptimize(
				model;
				SearchRange=$search_range,
				MaxTime=10.0
			)
			result_blackbox = BlackBoxOptim.best_candidate(result_blackbox)

			result_annealing = Optim.optimize(
				model,
				$lower,
				$upper,
				$initial,
				Optim.SAMIN(),
				Optim.Options(iterations=10^6)
			)
			result_annealing = Optim.minimizer(result_annealing)
		end)
	end

	function global_optimisation(
		measured_signal::AbstractVector{<: Number},
		T1::Tuple{Real, Real},
		T2::Tuple{Real, Real},
		relB1::Real,
		# Parameters from here
		cycles::Integer,
		kmax::Integer,
		α::AbstractVector{<: Real},
		ϕ::AbstractVector{<: Real},
		TR::Union{Tuple{Real, Real}, AbstractVector{<: Real}},
		G::AbstractVector{<: Real},
		τ::AbstractVector{<: Real},
		D::Real
	)
		# Make to func::Function, parameters::Dict{String, Any}

		α = relB1 .* α
		model::Function = let measured_signal = abs.(measured_signal)'
			θ -> begin
				simulation = abs.(MRIEPG.driven_equilibrium(
					kmax, cycles,
					α, ϕ, TR,
					G, τ, D,
					(θ[1], θ[2])
				)[1])
				simulation ./= -norm(simulation)
				measured_signal * simulation
			end
		end

		@global_optimisation_run(
			[R1, R2], # Search range
			[R1[1], R2[1]], # Lower
			[R1[2], R2[2]], # Upper
			[mean(R1), mean(R2)] # Initial parameters
		)

		return 1 ./ result_blackbox, 1 ./ result_annealing
	end

	function global_optimisation(
		measured_signal::AbstractVector{<: Number},
		T1::Tuple{Real, Real},
		T2::Tuple{Real, Real},
		relB1::Tuple{Real, Real},
		# Parameters from here
		cycles::Integer,
		kmax::Integer,
		α::AbstractVector{<: Real},
		ϕ::AbstractVector{<: Real},
		TR::Union{Tuple{Real, Real}, AbstractVector{<: Real}},
		G::AbstractVector{<: Real},
		τ::AbstractVector{<: Real},
		D::Real
	)
		# Make to func::Function, parameters::Dict{String, Any}
		# TODO: Macro for whole optimisation function, which eats up the "run"
		# TODO: make model generated to make use of TR = Union...

		model::Function = let measured_signal = abs.(measured_signal)'
			θ -> begin
				simulation = abs.(MRIEPG.driven_equilibrium(
					kmax, cycles,
					θ[3] .* α, ϕ, TR,
					G, τ, D,
					(θ[1], θ[2])
				)[1])
				simulation ./= -norm(simulation)
				measured_signal * simulation
			end
		end

		@global_optimisation_run(
			[R1, R2, relB1], # Search range
			[R1[1], R2[1], relB1[1]], # Lower
			[R1[2], R2[2], relB1[2]], # Upper
			[mean(R1), mean(R2), mean(relB1)] # Initial parameters
		)

		return (
			1 ./ result_blackbox[1:2],
			1 ./ result_annealing[1:2],
			result_blackbox[3],
			result_annealing[3]
		)
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
		# Make to func::Function, parameters::Dict{String, Any}

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

end

