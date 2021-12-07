
module MRFingerprinting

	import Optim
	using BlackBoxOptim

	@generated function global_optimisation_run(
		fitfunc::Function,
		measured_signal::AbstractArray{<: Number},
		T1::Tuple{Real, Real},
		T2::Tuple{Real, Real},
		relB1::Union{Real, Tuple{Real, Real}}
	)

		if relB1 <: Real
			search_range = :([R1, R2])
			lower = :([R1[1], R2[1]])
			upper = :([R1[2], R2[2]])
			initial = :(0.5 .* [sum(R1), sum(R2)])
			return_value = :(1 ./ result1, 1 ./ result2)
		else
			search_range = :([R1, R2, relB1])
			lower = :([R1[1], R2[1], relB1[1]])
			upper = :([R1[2], R2[2], relB1[2]])
			initial = :(0.5 .* [sum(R1), sum(R2), sum(relB1)])
			return_value = :(1 ./ result1[1:2], 1 ./ result2[1:2], result1[3], result2[3])
		end

		return quote
			# Put boundaries in Relaxometry
			R1 = 1 ./ reverse(T1)
			R2 = 1 ./ reverse(T2)

			result1 = bboptimize(
				fitfunc;
				SearchRange=$search_range,
				MaxTime=10.0
			)
			result1 = best_candidate(result1)

			result2 = Optim.optimize(
				fitfunc,
				$lower,
				$upper,
				$initial,
				Optim.SAMIN(),
				Optim.Options(iterations=10^6)
			)
			result2 = Optim.minimizer(result2)
			return $return_value
		end
	end

	function global_optimisation(
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

		α = relB1 .* α
		ms = abs.(measured_signal) # This is bogus
		fitfunc = R -> begin
			s = abs.(driven_equilibrium(
				α, ϕ, TR,
				kmax, cycles,
				G, τ, D,
				R[1],
				R[2]
			)[1])
			s ./= -norm(s)
			s' * ms
		end

		results = global_optimisation_run(fitfunc, measured_signal, T1, T2, relB1)
		return results
	end

	function global_optimisation(
		measured_signal::AbstractVector{<: Number},
		T1::Tuple{Real, Real},
		T2::Tuple{Real, Real},
		relB1::Tuple{Real, Real};
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
		fitfunc = R -> begin
			s = abs.(driven_equilibrium(
				R[3] .* α, ϕ, TR,
				kmax, cycles,
				G, τ, D,
				R[1],
				R[2]
			)[1])
			s ./= -norm(s)
			s' * ms
		end

		results = global_optimisation_run(fitfunc, measured_signal, T1, T2, relB1)
		return results
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

