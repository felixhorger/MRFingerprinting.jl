
#=
	If the schedule length and the shape of k-space are low integer multiples of each other,
	effort for optimising the sampling can be reduced by going block by block in k-space.
	This function extends the sampling optimised on a block onto the whole k-space
=#
function extend_sampling(
	optimised_sampling::AbstractVector{<: CartesianIndex{2}},
	num_σ::Integer,
	shape::NTuple{N, Integer},
	tiles::NTuple{N, Integer}
) where N
	num_time = length(optimised_sampling)
	num_patterns = num_time ÷ num_σ
	sampling_patterns = vec(MRIRecon.split_sampling_spatially(optimised_sampling, (num_patterns, 1), num_time))
	tile_size = shape .÷ tiles
	tile_length = prod(tile_size)
	@assert tile_length == length(sampling_patterns)
	tile_sampling_length = num_time

	sampling = Vector{CartesianIndex{2}}(undef, num_σ * prod(shape))
	sampling_tile = Array{Vector{Int}, 2}(undef, tile_size)
	inside_tile_indices = collect(CartesianIndices(tile_size))
	i = 1
	for I in CartesianIndices(tiles)
		I = CartesianIndex((Tuple(I) .- 1) .* tile_size)
		for (j, J) = enumerate(shuffle!(inside_tile_indices))
			sampling_tile[J] = sampling_patterns[j]
		end
		sampling[i:i+tile_sampling_length-1] = [I + J for J in MRIRecon.in_chronological_order(sampling_tile, num_time)]
		i += tile_sampling_length
	end
	return sampling
end



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

function improve_sampling!(
	split_sampling::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}},
	VH::AbstractMatrix{<: T},
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
		@views cond2 = cond(B2[:, 1:i2-1])
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

#function improve_sampling_parallel(
#	VH::AbstractMatrix{<: T},
#	split_sampling::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}},
#	shape::NTuple{N, Integer},
#	iter::Integer
#) where {N, T <: Number}
#	# Split into k-space locations, each index holds variable size vector of time indices
#	# Compute initial conditioning
#end


function improve_sampling_2!(
	split_sampling::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}},
	VH::AbstractMatrix{<: T},
	shape::NTuple{N, Integer},
	iter::Integer
) where {N, T <: Number}
	# Setup
	num_dynamic = length(split_sampling)
	i = 0 # iteration index
	replaced = 0

	sampling = MRIRecon.split_sampling_spatially(MRIRecon.in_chronological_order(split_sampling), shape, num_dynamic)


	# Iterate
	while i < iter

		# Find minimum and maximum eigenvalue
		λ = compute_eigenvalues(VH, shape, sampling)
		conditioning, i_mini, i_maxi = calculate_conditioning(λ)

		# Swap something involving i_mini and i_maxi
		times_mini = sampling[i_mini]
		times_maxi = sampling[i_maxi]

		while i < iter
			i += 1
			# Pick random times
			j = rand((1, 2))
			times = (times_mini, times_maxi)[j]
			t1 = rand(1:length(times))
			s1 = (i_mini, i_maxi)[j]
			# Pick random samples
			s2 = rand(CartesianIndices(shape))
			s1 == s2 && continue
			t2 = rand(1:length(sampling[s2]))
			(sampling[s1][t1] in sampling[s2] || sampling[s2][t2] in sampling[s1]) && continue

			# Swap
			sampling[s1][t1], sampling[s2][t2] = sampling[s2][t2], sampling[s1][t1]

			# Find minimum and maximum eigenvalue
			λ[s1, :] .= eigvals(lowrank_mixing(VH, sampling[s1]))
			λ[s2, :] .= eigvals(lowrank_mixing(VH, sampling[s2]))
			swapped_conditioning, swapped_i_mini, swapped_i_maxi = calculate_conditioning(λ)
			@show swapped_conditioning, conditioning

			if swapped_conditioning < conditioning
				replaced += 1
				break
			else
				sampling[s1][t1], sampling[s2][t2] = sampling[s2][t2], sampling[s1][t1]
			end
		end
	end
	return MRIRecon.split_sampling(sampling, num_dynamic), replaced
end

# singular values
function compute_eigenvalues(
	VH::AbstractMatrix{<: T},
	shape::NTuple{N, Integer}, # TODO: wth is this here, since shape == size(sampling)?
	sampling::AbstractArray{<: AbstractVector{<: Integer}, N}
) where {N, T <: Number}
	# TODO: limitation is that number of samples must be the same for every k
	# Compute eigenvalues
	num_σ, num_dynamic = size(VH)
	λ = Array{real(T), N+1}(undef, shape..., num_σ)
	lr_mix = Matrix{T}(undef, num_σ, maximum(length, sampling)) # This is not really lr_mix, just L or L^H, rename
	for I in CartesianIndices(shape)
		if length(sampling) == 0
			λ[I, :] .= 0
		end
		for (i, t) in enumerate(sampling[I])
			@views lr_mix[:, i] = VH[:, t]
		end
		j = length(sampling[I])
		@views λ[I, 1:min(j, num_σ)] = svd!(qr(lr_mix[:, 1:j], LinearAlgebra.ColumnNorm()).R; alg=LinearAlgebra.QRIteration()).S
		@views λ[I, j+1:num_σ] .= 0
	end
	return λ
end
# Actual eigenvalues
#function compute_eigenvalues(
#	VH::AbstractMatrix{<: T},
#	shape::NTuple{N, Integer},
#	sampling::AbstractArray{<: AbstractVector{<: Integer}, N}
#) where {N, T <: Number}
#	# Compute eigenvalues
#	num_σ, num_dynamic = size(VH)
#	λ = Array{real(T), N+1}(undef, shape..., num_σ)
#	for I in CartesianIndices(shape)
#		lr_mix = lowrank_mixing(VH, sampling[I])
#		@views λ[I, :] = abs.(eigvals!(Hermitian(lr_mix)))
#	end
#	return λ
#end

function calculate_conditioning_parallel(λ::AbstractArray{<: Number, N}) where N
	shape = size(λ)[1:N-1]
	num_σ = size(λ, N)
	mini = λ[1]
	i_mini = one(CartesianIndex{N-1})
	maxi = mini
	i_maxi = one(CartesianIndex{N-1})
	thread_mini = fill(mini, Threads.nthreads())
	thread_maxi = fill(maxi, Threads.nthreads())
	thread_i_mini = fill(i_mini, Threads.nthreads())
	thread_i_maxi = fill(i_maxi, Threads.nthreads())
	@inbounds Threads.@threads :static for I in CartesianIndices(shape)
		tid = Threads.threadid()
		λ_min = λ[I, num_σ]
		λ_max = λ[I, 1]
		if thread_mini[tid] > λ_min
			thread_i_mini[tid] = I
			thread_mini[tid] = λ_min
		elseif thread_maxi[tid] < λ_max
			thread_i_maxi[tid] = I
			thread_maxi[tid] = λ_max
		end
	end
	mini, j_mini = findmin(thread_mini)
	i_mini = thread_i_mini[j_mini]
	maxi, j_maxi = findmax(thread_maxi)
	i_maxi = thread_i_maxi[j_maxi]
	conditioning = maxi / mini
	return conditioning, i_mini, i_maxi, mini, maxi
end

# eigenvalues must be sorted!
function calculate_conditioning(λ::AbstractArray{<: Number, N}) where N
	shape = size(λ)[1:N-1]
	num_σ = size(λ, N)
	mini = λ[1]
	i_mini = one(CartesianIndex{N-1})
	maxi = mini
	i_maxi = one(CartesianIndex{N-1})
	@inbounds for I in CartesianIndices(shape)
		λ_min = λ[I, num_σ] # TODO: adpated to SVD
		λ_max = λ[I, 1]
		if mini > λ_min
			i_mini = I
			mini = λ_min
		end
		if maxi < λ_max
			i_maxi = I
			maxi = λ_max
		end
	end
	conditioning = maxi / mini
	return conditioning, i_mini, i_maxi, mini, maxi
end




function improve_sampling_3!(
	split_sampling::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}},
	VH::AbstractMatrix{<: T},
	shape::NTuple{N, Integer},
	iter::Integer;
	maxiter::Integer=0 # zero means search all
) where {N, T <: Number}
	num_σ, num_dynamic = size(VH)

	# Split into spatial indices
	sampling = MRIRecon.split_sampling_spatially(MRIRecon.in_chronological_order(split_sampling), shape, num_dynamic)

	# Compute initial eigenvalues and conditioning
	λ = compute_eigenvalues(VH, shape, sampling)
	conditioning, i_mini, i_maxi, λ_min, λ_max = calculate_conditioning(λ)
	#@show "before", Threads.threadid(), conditioning

	# Look-up for spatial indices
	cartesian_indices = CartesianIndices(shape)

	# Pre-allocate some memory
	num_spatial = prod(shape)
	linear_indices = Vector{Int}(undef, 2num_spatial)
	# Threaded
	λ_1 = Vector{real(T)}(undef, num_σ)
	λ_2 = Vector{real(T)}(undef, num_σ)
	#random_two = Vector{Int}(undef, 2)
	random_t1 = [Vector{Int}(undef, t) for t = 1:num_dynamic]
	random_t2 = [Vector{Int}(undef, t) for t = 1:num_dynamic]

	if maxiter == 0
		maxiter = 2num_spatial
	else
		maxiter = min(maxiter, 2num_spatial)
	end

	lr_mix = Matrix{T}(undef, num_σ, length(sampling[1]))

	# Iterate
	i = 0
	@inbounds while i < iter
		#mod(i, 1000) == 0 && @show i, conditioning
		#λ = compute_eigenvalues(VH, shape, sampling)
		#@show i, conditioning

		#randperm!(random_two)
		found = false
		for (progress, m) in @views enumerate(randperm!(linear_indices)[1:maxiter])
			#mod(progress, 1000) == 0 && @show round(progress / maxiter, digits=2)
			# Get random indices
			j = mod1(m, 2) # random_two[I[1]]
			k = (m-1) ÷ 2 + 1 # CartesianIndex(Tuple(I)[2])
			s1 = (i_mini, i_maxi)[j]
			s2 = cartesian_indices[k]
			s1 == s2 && continue
			times1 = sampling[s1]
			times2 = sampling[s2]
			for t1 in randperm!(random_t1[length(times1)])
				# Is t1 in already in the times sampled at s2?
				times1[t1] in times2 && continue
				for t2 in randperm!(random_t2[length(times2)])
					# Is t2 already in the times sampled at s1?
					times2[t2] in times1 && continue

					# Swap
					times1[t1], times2[t2] = times2[t2], times1[t1]

					# Find minimum and maximum eigenvalue
					#λ_new_1 = abs.(eigvals!(Hermitian(lowrank_mixing(VH, times1))))
					#λ_new_2 = abs.(eigvals!(Hermitian(lowrank_mixing(VH, times2))))
					for (r, t) in enumerate(times1)
						@views lr_mix[:, r] = VH[:, t]
					end
					λ_new_1 = svd!(qr(lr_mix, LinearAlgebra.ColumnNorm()).R; alg=LinearAlgebra.QRIteration()).S
					for (r, t) in enumerate(times2)
						@views lr_mix[:, r] = VH[:, t]
					end
					λ_new_2 = svd!(qr(lr_mix, LinearAlgebra.ColumnNorm()).R; alg=LinearAlgebra.QRIteration()).S

					# Are new min/max eigenvalues smaller/greater than old ones?
					if min(λ_new_1[end], λ_new_2[end]) ≤ λ_min && max(λ_new_1[1], λ_new_2[1]) ≥ λ_max
						# Swap back
						times1[t1], times2[t2] = times2[t2], times1[t1]
						continue
					end

					# Need to determine new min/max eigenvalues and then check if conditioning improved

					# Save old eigenvalues
					@views λ_1 .= λ[s1, :]
					@views λ_2 .= λ[s2, :]

					# Put new ones
					@views λ[s1, :] = λ_new_1
					@views λ[s2, :] = λ_new_2

					# Determine new conditioning
					(
						swapped_conditioning,
						swapped_i_mini, swapped_i_maxi,
						swapped_λ_mini, swapped_λ_maxi
					) = calculate_conditioning(λ)

					# If conditioning improved, keep swap, otherwise restore old values
					if swapped_conditioning < conditioning
						#@show i, swapped_conditioning, conditioning, λ_min, λ_max, i_mini, i_maxi, swapped_λ_mini, swapped_λ_maxi
						#@show swapped_conditioning, conditioning
						conditioning = swapped_conditioning                                        
						i_mini = swapped_i_mini
						i_maxi = swapped_i_maxi
						λ_min = swapped_λ_mini
						λ_max = swapped_λ_maxi
						found = true
						break
					else
						@views λ[s1, :] = λ_1
						@views λ[s2, :] = λ_2
						times1[t1], times2[t2] = times2[t2], times1[t1]
					end
				end
				found && break
			end
			found && break
		end
		# If nothing was found, the sampling pattern can't be further optimised
		!found && break
		i += 1
	end
	#sampling = MRIRecon.split_sampling(sampling, num_dynamic)
	#sampling = MRIRecon.split_sampling_spatially(MRIRecon.in_chronological_order(sampling), shape, num_dynamic)
	#new_λ = compute_eigenvalues(VH, shape, sampling)
	#(
	#	swapped_conditioning,
	#	swapped_i_mini, swapped_i_maxi,
	#	swapped_λ_mini, swapped_λ_maxi
	#) = calculate_conditioning(new_λ)
	#@show i, conditioning, swapped_conditioning
	#@show λ_min, swapped_λ_mini
	#@show λ_max, swapped_λ_maxi
	#@show i_mini, swapped_i_mini
	#@show i_maxi, swapped_i_maxi
	#@assert isapprox(new_λ, λ)
	#@assert swapped_conditioning == conditioning
	#@assert swapped_i_mini == i_mini
	#@assert swapped_i_maxi == i_maxi
	#@assert swapped_λ_mini == λ_min
	#@assert swapped_λ_maxi == λ_max

	#λ = compute_eigenvalues(VH, shape, sampling)
	#@show (
	#	swapped_conditioning,
	#	swapped_i_mini, swapped_i_maxi,
	#	swapped_λ_mini, swapped_λ_maxi
	#) = calculate_conditioning(λ)

	#sampling2 = copy(sampling)
	#sampling = MRIRecon.split_sampling_spatially(MRIRecon.in_chronological_order(MRIRecon.split_sampling(sampling, num_dynamic)), shape, num_dynamic)
	#for I in CartesianIndices(shape)
	#	@assert sort(sampling2[I]) == sort(sampling[I])
	#end


	#λ = compute_eigenvalues(VH, shape,
	#MRIRecon.split_sampling_spatially(
	#	MRIRecon.in_chronological_order(
	#		MRIRecon.split_sampling(sampling, num_dynamic)), shape, num_dynamic))


	#@show (
	#	swapped_conditioning,
	#	swapped_i_mini, swapped_i_maxi,
	#	swapped_λ_mini, swapped_λ_maxi
	#) = calculate_conditioning(λ)
	#@show "after", Threads.threadid(), conditioning
	return MRIRecon.split_sampling(sampling, num_dynamic), conditioning, i
end

function improve_sampling_4!(
	split_sampling::AbstractVector{<: AbstractVector{<: CartesianIndex{N}}},
	VH::AbstractMatrix{<: T},
	shape::NTuple{N, Integer},
	iter::Integer
) where {N, T <: Number}
	num_σ, num_dynamic = size(VH)

	# Split into spatial indices
	sampling = MRIRecon.split_sampling_spatially(MRIRecon.in_chronological_order(split_sampling), shape, num_dynamic)

	# Look-up for spatial indices
	cartesian_indices = CartesianIndices(shape)

	num_spatial = prod(shape)
	num_threads = Threads.nthreads()
	linear_indices = Vector{Int}(undef, num_spatial)
	all_indices_to_iterate = CartesianIndices((num_spatial, 2))
	thread_indices_to_iterate = @views [
		all_indices_to_iterate[range(ThreadTools.thread_region(tid, length(all_indices_to_iterate), num_threads)...)]
		for tid = 1:num_threads
	]

	λ_1 = [Vector{Float64}(undef, num_σ) for tid = 1:num_threads]
	λ_2 = [Vector{Float64}(undef, num_σ) for tid = 1:num_threads]
	thread_times1 = [Vector{Int}(undef, num_dynamic) for tid = 1:num_threads]
	thread_times2 = [Vector{Int}(undef, num_dynamic) for tid = 1:num_threads]
	random_two = Vector{Int}(undef, 2)
	random_t1 = [ [Vector{Int}(undef, t) for t = 1:num_dynamic] for tid = 1:num_threads ]
	random_t2 = [ [Vector{Int}(undef, t) for t = 1:num_dynamic] for tid = 1:num_threads ]

	# Compute initial eigenvalues and conditioning
	λs = [Array{Float64}(undef, shape..., num_σ) for tid = 1:num_threads]
	λs[1] = compute_eigenvalues(VH, shape, sampling)
	conditioning, i_mini, i_maxi, λ_min, λ_max = calculate_conditioning(λs[1])

	# Iterate
	i = 0 # iteration index
	swapped = 0 # counter for how many points have been swapped
	found_tid = 1
	found_lock = ReentrantLock()
	threads_checkout = Base.Semaphore(num_threads)
	foreach(tid -> Base.acquire(threads_checkout), 1:num_threads)
	debug_step = max(iter ÷ 100, 1)
	while i < iter
		mod(i, debug_step) == 0 && @show i, conditioning

		found = false

		for tid = 1:num_threads
			tid == found_tid && continue
			λs[tid] .= λs[found_tid]
		end

		randperm!(linear_indices)
		randperm!(random_two)

		@sync @inbounds for tid = 1:num_threads
			Threads.@spawn let	λ_1=λ_1[tid], λ_2=λ_2[tid], λ=λs[tid],
								random_t1=random_t1[tid], random_t2=random_t2[tid],
								thread_times1=thread_times1[tid], thread_times2=thread_times2[tid]

				for I in thread_indices_to_iterate[tid]

					if islocked(found_lock)
						Base.release(threads_checkout)
						#@show tid, "checkout"
						break
					end

					# Get random indices
					k = linear_indices[I[1]]
					j = random_two[I[2]]
					# Get 
					s1 = (i_mini, i_maxi)[j]
					s2 = cartesian_indices[k]
					s1 == s2 && continue
					times1 = @view thread_times1[1:length(sampling[s1])]
					times2 = @view thread_times2[1:length(sampling[s2])]
					times1 .= sampling[s1] # TODO: this is unnecessary copying, bc s1 is more or less fixed (see j)
					times2 .= sampling[s2]
					rand_t1 = randperm!(random_t1[length(times1)])
					rand_t2 = randperm!(random_t2[length(times2)])

					for t1 in rand_t1
						islocked(found_lock) && break
						# Is t1 in already in the times sampled at s2?
						times1[t1] in times2 && continue
						for t2 in rand_t2
							islocked(found_lock) && break
							# Is t2 already in the times sampled at s1?
							times2[t2] in times1 && continue

							# Swap
							times1[t1], times2[t2] = times2[t2], times1[t1]

							# Find minimum and maximum eigenvalue
							λ_new_1 = eigvals!(Hermitian(lowrank_mixing(VH, times1)))
							λ_new_2 = eigvals!(Hermitian(lowrank_mixing(VH, times2)))

							# Are new min/max eigenvalues smaller/greater than old ones?
							if min(λ_new_1[1], λ_new_2[1]) ≤ λ_min && max(λ_new_1[end], λ_new_2[end]) ≥ λ_max
								# Swap back
								times1[t1], times2[t2] = times2[t2], times1[t1]
								continue
							end

							# Need to determine new min/max eigenvalues and then check if conditioning improved

							# Save old eigenvalues
							@views λ_1 .= λ[s1, :]
							@views λ_2 .= λ[s2, :]

							# Put new ones
							@views λ[s1, :] = λ_new_1
							@views λ[s2, :] = λ_new_2

							# Determine new conditioning
							(
								swapped_conditioning, swapped_i_mini, swapped_i_maxi,
								swapped_λ_min, swapped_λ_max
							) = calculate_conditioning(λ)

							# If conditioning improved, keep swap, otherwise restore old values
							if swapped_conditioning < conditioning
								!trylock(found_lock) && break
								found = true
								found_tid = tid
								#@show tid, "asdasdasd"
								Base.release(threads_checkout)
								for _ = 1:num_threads 
									Base.acquire(threads_checkout)
								end
								unlock(found_lock)
								# Update
								conditioning = swapped_conditioning
								i_mini = swapped_i_mini
								i_maxi = swapped_i_maxi
								λ_min = swapped_λ_min
								λ_max = swapped_λ_max
								sampling[s1] = times1[1:length(sampling[s1])]
								sampling[s2] = times2[1:length(sampling[s2])]
								return
							else
								@views λ[s1, :] = λ_1
								@views λ[s2, :] = λ_2
								times1[t1], times2[t2] = times2[t2], times1[t1]
							end
						end
					end
				end
			end
		end
		#@show "all restart"

		# If nothing was found, the sampling pattern can't be further optimised
		!found && break
		i += 1
	end
	return MRIRecon.split_sampling(sampling, num_dynamic), conditioning, i
end

