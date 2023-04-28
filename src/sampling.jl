
function improve_sampling(
	VH::AbstractMatrix{<: T},
	split_sampling::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}},
	shape::NTuple{N, Integer},
	iter::Integer
) where {N, T <: Number}
	# Compute initial conditioning
	num_σ, num_dynamic = size(VH)
	lr_mix = lowrank_mixing(VH, MRIRecon.in_chronological_order(split_sampling), shape)
	conditioning = Array{T, N}(undef, shape)
	for I in CartesianIndices(shape)
		conditioning[I] = @views sqrt(cond(lr_mix[I, :, :])) # √ because lr mix has the L^H U L, but below only U L is used
	end
	# Setup
	i = 0 # iteration index
	# Mixing matrices (one half actually: U L)
	# Just allocate maximum possible space requirement
	M1 = Matrix{Float64}(undef, num_σ, num_dynamic)
	M2 = Matrix{Float64}(undef, num_σ, num_dynamic)
	replaced = 0
	total_gain = 0.0
	# Iterate
	@inbounds while i < iter
		# Pick random times
		t1, t2 = mod1.(rand(Int, 2), num_dynamic)
		t1 == t2 && continue
		# Pick random samples
		n1, n2 = mod1.(rand(Int, 2), (length(split_sampling[t1]), length(split_sampling[t2])))
		s1 = split_sampling[t1][n1]
		s2 = split_sampling[t2][n2]
		(s1 in split_sampling[t2] || s2 in split_sampling[t1]) && continue
		# Counters for how many samples there are
		i1 = 1
		i2 = 1
		# Collect all num_σ-elements vectors at k-space locations s1 and s2
		@views for t = 1:num_dynamic
			v = VH[:, t] # Current num_σ-elements vector
			if s1 in split_sampling[t]
				if t == t1 # This is to be swapped, thus assign to other matrix
					M2[:, i2] = v
					i2 += 1
				else # Not to be swapped, assign to the same matrix
					M1[:, i1] = v
					i1 += 1
				end
			end
			# Analogous for s2
			if s2 in split_sampling[t]
				if t == t2
					M1[:, i1] = v
					i1 += 1
				else
					M2[:, i2] = v
					i2 += 1
				end
			end
		end
		# Compute gain = old conditioning / new conditioning
		gain = @views (conditioning[s1] / cond(M1[:, 1:i1-1])) * (conditioning[s2] / cond(M2[:, 1:i2-1]))
		if gain > 1
			# Swap
			split_sampling[t1][n1], split_sampling[t2][n2] = split_sampling[t2][n2], split_sampling[t1][n1]
			# Update stats
			replaced += 1
			total_gain += gain
		end
		i += 1
	end
	return split_sampling, replaced, total_gain
end

