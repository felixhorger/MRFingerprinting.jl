
function best_temporal_sampling(t0::Integer, num_samples::Integer, VH::AbstractMatrix{<: T}) where T <: Number
	num_σ, num_dynamic = size(VH)
	B = Matrix{real(T)}(undef, num_σ, num_samples)
	M = Matrix{real(T)}(undef, num_σ, num_σ)
	M_hermitian = Hermitian(M)
	t = Vector{Int64}(undef, num_samples)
	t[1] = t0
	@views B[:, 1] = VH[:, t0]
	local current_conditioning = 1
	for i = 2:num_samples
		t_next = 1
		current_conditioning = Inf
		for τ = 1:num_dynamic
			@views B[:, i] = VH[:, τ]
			@views mul!(M, B[:, 1:i], B[:, 1:i]')
			κ = cond(M_hermitian)
			if κ < current_conditioning
				current_conditioning = κ
				t_next = τ
			end
		end
		t[i] = t_next
		@views B[:, i] = VH[:, t_next]
	end
	return t, current_conditioning, B, M
end

function improve_sampling(
	VH::AbstractMatrix{<: T},
	split_sampling::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}},
	shape::NTuple{N, Integer},
	iter::Integer
) where {N, T <: Number}
	# Compute initial conditioning
	num_σ, num_dynamic = size(VH)
	lr_mix = lowrank_mixing(VH, MRIRecon.in_chronological_order(split_sampling), shape)
	conditioning = Array{real(T), N}(undef, shape)
	for I in CartesianIndices(shape)
		conditioning[I] = @views sqrt(cond(lr_mix[I, :, :])) # √ because lr mix has the L^H U L, but below only U L is used
	end
	# Setup
	i = 0 # iteration index
	# Mixing matrices (B = U L, M = L^H U L)
	# Just allocate maximum possible space requirement
	B1 = Matrix{T}(undef, num_σ, num_dynamic)
	B2 = Matrix{T}(undef, num_σ, num_dynamic)
	M = zeros(T, num_σ, num_σ)
	M_hermitian = Hermitian(M)
	replaced = 0
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
					B2[:, i2] = v
					i2 += 1
				else # Not to be swapped, assign to the same matrix
					B1[:, i1] = v
					i1 += 1
				end
			end
			# Analogous for s2
			if s2 in split_sampling[t]
				if t == t2
					B1[:, i1] = v
					i1 += 1
				else
					B2[:, i2] = v
					i2 += 1
				end
			end
		end
		# Compute gain = old conditioning / new conditioning
		#if conditioning[s1] > 1e15 || conditioning[s2] > 1e15 # Singular?
		#	# Have to use the full mixing matrix, check if still singular
		#	@views mul!(M, B1[:, 1:i1-1], B1[:, 1:i1-1]')
		#	cond1 = cond(M_hermitian)
		#	@views mul!(M, B2[:, 1:i1-1], B2[:, 1:i1-1]')
		#	cond2 = cond(M_hermitian)
		#	if cond1 < 1e15 && cond2 < 1e15
		#		# Both became non-singular, keep this
		#		gain = 1.1 # anything larger than 1
		#	else
		#		gain = 0
		#	end
		# dont forget sqrt
		#end
		@views cond1 = cond(B1[:, 1:i1-1])
		@views cond2 = cond(B2[:, 1:i1-1])
		gain = @views (conditioning[s1] / cond1) * (conditioning[s2] / cond2)
		if gain > 1
			# Swap
			split_sampling[t1][n1], split_sampling[t2][n2] = split_sampling[t2][n2], split_sampling[t1][n1]
			# Change conditioning
			conditioning[s1] = cond1
			conditioning[s2] = cond2
			# Update stats
			replaced += 1
		end
		i += 1
	end
	return split_sampling, replaced
end

function improve_sampling_parallel(
	VH::AbstractMatrix{<: T},
	split_sampling::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}},
	shape::NTuple{N, Integer},
	iter::Integer
) where {N, T <: Number}
	# Split into k-space locations, each index holds variable size vector of time indices
	# Compute initial conditioning
end

