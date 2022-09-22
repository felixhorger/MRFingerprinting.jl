
# Definitions: Use flat spatial dimensions, i.e. vector is a matrix with one spatial and one temporal dimension.
# For all the CG stuff however, this needs to be reshaped into a single vector

"""
	Useful for compression of actual kspace data for creating spatial ft without having to do the reshaping every time
	Of course works for any time domain array
"""
function time2lr(xt::AbstractArray{<: Number, N}, V_conj::AbstractMatrix{<: Number}) where N
	# TODO: This could be done with @turbo
	shape = size(xt)
	xt = reshape(xt, :, shape[N])
	x = xt * V_conj
	x = reshape(x, shape[1:N-1]..., size(V_conj, 2))
	return x
end
"""
	Sparse version of time2lr, so made for kspace only
"""
function kt2klr(
	x::AbstractArray{<: Number, 3},
	VH::AbstractMatrix{<: Number},
	shape::NTuple{N, Integer},
	indices::AbstractVector{<: CartesianIndex{N}}
) where N
	@assert all(size(x, 3) .== length.(indices))
	@assert all(all(0 .< indices[d] .<= shape[d]) for d ∈ 1:N)
	timepoints = size(VH, 2)
	y = zeros(ComplexF64, size(x, 1), shape..., size(x, 2), size(VH, 1)) # spatial dims..., channels, sigma
	# Must be zeros because if a location isn't sampled it would contain undefs
	@inbounds for (i, j) in enumerate(indices)
		t = mod1(i, timepoints)
		for σ in axes(VH, 1)
			y[:, j, :, σ] += VH[σ, t] * x[:, :, i]
		end
	end
	return y
end


"""
	 Convenience for getting the more practical versions of V: V* and V^T
"""
function convenient_Vs(VH::AbstractMatrix{<: Number})
	collect(transpose(VH)), conj.(VH)
end


"""
	num_other is everything except from dynamic
"""
function plan_lr2time(V_conj::AbstractMatrix{<: Number}, VT::AbstractMatrix{<: Number}, num_other::Integer)
	# TODO: This could be done with @turbo
	num_time, num_σ = size(V_conj)
	@assert size(VT) == (num_σ, num_time)
	input_dimension = num_σ * num_other
	output_dimension = num_time * num_other
	Λ = LinearMap{ComplexF64}(
		y -> begin
			y = reshape(y, num_other, num_σ)
			yt = y * VT
			vec(yt)
		end,
		yt -> begin
			yt = reshape(yt, num_other, num_time)
			y = yt * V_conj
			vec(y)
		end,
		output_dimension,
		input_dimension
	)
	return Λ
end






"""
	plan_lr2kt(L::LinearMap, F::LinearMap)

"""
@inline plan_lr2kt(Λ::LinearMap, F::LinearMap) = Λ * F
"""
	plan_lr2kt(L::LinearMap, F::LinearMap, S::LinearMap)

"""
@inline plan_lr2kt(Λ::LinearMap, F::LinearMap, S::LinearMap) = plan_lr2kt(Λ, F) * S



"""
	shape is the shape of the phase encoding plane
	VH[singular component, time]
	
	For each voxel, mask singular vectors in V,
	then do inner product with vectors in VH to form one matrix per voxel,
	transforming signals from the temporal low-rank domain into itself.

	indices must be unique in the k-t domain otherwise this will be wrong!
"""
function lowrank_mixing(
	VT::AbstractMatrix{<: T},
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer}
) where {N, T}
	num_σ, num_dynamic = size(VT)
	linear_indices = LinearIndices(shape)
	perm = sortperm(indices; by=(x::CartesianIndex{N} -> linear_indices[x]))
	lr_mix = zeros(T, num_σ, num_σ, shape...)
	for i in eachindex(perm)
		j = perm[i]
		dynamic = mod1(j, num_dynamic)
		k = indices[j]
		# Outer product of singular vectors
		# M_{σ,σ'}	= ∑_{t,t'} V^H_{σ,t} ⋅ U_{t,t'} ⋅ V_{t',σ'}
		#			= ∑_{t,t'} U_{t,t'} ⋅ V_{t',σ'} ⋅ V^*_{t,σ}
		#			= ∑_{t,t'} u_t ⋅ δ_{t,t'} ⋅ V_{t',σ'} ⋅ V^*_{t,σ}
		#			= ∑_t u_t ⋅ V^*_{t,σ} ⋅ V_{t,σ'}
		for σ2 = 1:num_σ, σ1 = 1:num_σ
			#v = ...
			lr_mix[σ1, σ2, k] += conj(VT[σ1, dynamic]) * VT[σ2, dynamic]
			# Note: the mixing matrices are Hermitian, thus the above could be σ2 = σ1:num_σ
			# and then copying the other half:
			#if σ1 ≠ σ2
			#	lr_mix[σ2, σ1, k] += conj(v)
			#end
			# However it doesn't pay off if num_σ is small
		end
	end
	return lr_mix
end

"""

lr_mix[σ, σ, spatial dimensions]

"""
function apply_lowrank_mixing(
	y::AbstractVector{C},
	lr_mix::AbstractArray{<: Real, 3}, # Flat spatial dimension
	ι::Integer,
	κ::Integer
) where C <: Complex
	num_σ = size(lr_mix, 1)
	num_x = size(lr_mix, 3)
	# TODO: check shape
	y = reshape(y, ι, num_x, κ, num_σ)
	yd = decomplexify(y)
	ymd = similar(yd) # y *m*ixed and *d*ecomplexified
	@turbo for k = 1:κ, x = 1:num_x, i = 1:ι
		for σ2 = 1:num_σ
			ym_real = 0.0
			ym_imag = 0.0
			for σ1 = 1:num_σ
				ym_real += yd[1, i, x, k, σ1] * lr_mix[σ1, σ2, x]
				ym_imag += yd[2, i, x, k, σ1] * lr_mix[σ1, σ2, x]
			end
			ymd[1, i, x, k, σ2] = ym_real
			ymd[2, i, x, k, σ2] = ym_imag
		end
	end
	ym = reinterpret(C, vec(ymd))
	return ym
end
function apply_lowrank_mixing(
	y::AbstractVector{C},
	lr_mix_d::AbstractArray{<: Real, 4},
	ι::Integer,
	κ::Integer
) where C <: Complex
	num_σ = size(lr_mix_d, 3)
	num_x = size(lr_mix_d, 4)
	y = reshape(y, readout_length, num_x, channels, num_σ)
	yd = decomplexify(y)
	ymd = similar(yd) # y *m*ixed and *d*ecomplexified
	@tturbo for k = 1:κ, x = 1:num_x, i = 1:ι
		for σ2 = 1:num_σ
			ym_real = 0.0
			ym_imag = 0.0
			for σ1 = 1:num_σ
				ym_real += (
					  yd[1, i, x, k, σ1] * lr_mix_d[1, σ1, σ2, x]
					- yd[2, i, x, k, σ1] * lr_mix_d[2, σ1, σ2, x]
				)
				ym_imag += (
					  yd[1, i, x, k, σ1] * lr_mix_d[2, σ1, σ2, x]
					+ yd[2, i, x, k, σ1] * lr_mix_d[1, σ1, σ2, x]
				)
			end
			ymd[1, i, x, k, σ2] = ym_real
			ymd[2, i, x, k, σ2] = ym_imag
		end
	end
	ym = reinterpret(C, vec(ymd))
	return ym
end



"""

lr_mix[σ1, σ2, spatial dimensions...]
lr_mix not copied

ι,κ = number of elements in the first,last spatial dimension of y
which is assumed to be acquires at the same time.
For example: 3D Cartesian imaging would have dimensions (num_lines, num_partitions, num_readouts)
where readouts can be Fourier transformed beforehand. In this case ι = 1 and κ = num_readouts.
For stack of stars with fully sampled partition direction, (num_columns, num_lines, num_partitions)
is more favourable, so ι = num_columns and κ = num_partitions.
For multi-channel data, the number of channels and κ can be fused.

"""
function plan_lowrank_mixing(lr_mix::AbstractArray{<: Number, N}, ι::Integer, κ::Integer) where N
	# Get and check shapes
	lr_mix_shape = size(lr_mix)
	num_σ = lr_mix_shape[1]
	@assert lr_mix_shape[2] == num_σ
	shape = lr_mix_shape[3:N]
	num_phase_encode = prod(shape)
	# Reshape and split real/imag
	lr_mix = reshape(lr_mix, num_σ, num_σ, num_phase_encode) # It isn't copied
	lr_mix_d = decomplexify(lr_mix)
	# Define function
	M = LinearMap{ComplexF64}(
		y::AbstractVector{<: Complex} -> begin
			apply_lowrank_mixing(y, lr_mix_d, ι, κ)
		end,
		num_phase_encode * ι * κ * num_σ,
		ishermitian=true
	)
	return M
end
function plan_lowrank_mixing(
	VT::AbstractMatrix{<: Number},
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	ι::Integer,
	κ::Integer,
) where N
	lr_mix = lowrank_mixing(VT, indices, shape)
	return plan_lowrank_mixing(lr_mix, ι, κ)
end



function lowrank_sparse2dense(
	kspace::AbstractArray{<: T, 3}, # [readout, channel, phase encoding]
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	VH::AbstractMatrix{<: Number}
) where {N, T <: Number}
	num_σ, num_dynamic = size(VH)
	linear_indices = LinearIndices(shape)
	perm = sortperm(indices; by=(k::CartesianIndex{N} -> linear_indices[k]))
	backprojection = zeros(T, size(kspace, 1), size(kspace, 2), shape..., num_σ)
	for i in eachindex(perm)
		j = perm[i]
		dynamic = mod1(j, num_dynamic)
		k = indices[j]
		for σ = 1:num_σ
			@views backprojection[:, :, k, σ] += kspace[:, :, j] * VH[σ, dynamic]
		end
	end
	return backprojection
end



"""

x[spatial dimensions ..., singular components]

`matches` must be preallocated and filled to update P
DT_renorm[σ, fingerprints] must be normalised in the SVD domain after cut-off

"""
function projection_matrix(
	DT_renorm::AbstractMatrix{<: Real},
	matches::AbstractVector{<: Integer},
	num_x::Integer,
	num_σ::Integer,
	T::Type{C}
) where C <: Complex
	@assert length(matches) == num_x
	@assert size(DT_renorm, 1) == num_σ
	num_D = size(DT_renorm, 2)

	P = LinearMap{ComplexF64}(
		x::AbstractVector{<: Complex} -> begin
			@assert all((i -> 0 <= i <= num_D), matches) # Zero is a forbidden index, in that case a zero filled vector will be returned
			x = reshape(x, num_x, num_σ) # Weird: this has to be done before complexifying, otherwise the same error as in overlap!() with views
			xd = decomplexify(x) # d for decomplexified
			xpd = similar(xd)
			for xi in eachindex(matches)
				@inbounds match = matches[xi]
				p_real = 0.0
				p_imag = 0.0
				@turbo for σ = 1:num_σ # Shame, two loops at one level not supported by LoopVectorization
					p_real += DT_renorm[σ, match] * xd[1, xi, σ]
					p_imag += DT_renorm[σ, match] * xd[2, xi, σ]
				end
				@turbo for σ = 1:num_σ
					xpd[1, xi, σ] = p_real * DT_renorm[σ, match]
					xpd[2, xi, σ] = p_imag * DT_renorm[σ, match]
				end
			end
			xp = reinterpret(C, vec(xpd))
			xp
		end,
		num_x * num_σ,
		ishermitian=true
	)
	return P
end



"""
	plan_psf_regularised(n::Integer, A::LinearMap, P::LinearMap)
	A = S' * F' * M * F * S, i.e. the PSF without the projection.
	Note that M can be a low-rank mixing, i.e. this acts on a vector
	in temporal low-rank image space.

"""
function plan_psf_regularised(A::LinearMap, P::LinearMap, ρ::Real)
	Ar = LinearMap{ComplexF64}(
		(x -> A*x + ρ * (x - P*x)),
		size(A, 1),
		ishermitian=true
	)
	return Ar
end



"""

Creates closure with all relevant info for matching

"""
function matching_closure(
	D::AbstractMatrix{<: Number},
	indices::AbstractVector{<: Integer},
	stride::Integer,
	step::Integer,
	num_f::Integer
)
	# TODO: Could do this in a way that it writes into the matches array directly
	num_σ = size(D, 2)
	match(x::NTuple{1, AbstractMatrix{<: Number}}) = MRFingerprinting.match(D, x[1], indices, stride, step)
	match(x::NTuple{2, AbstractMatrix{<: Number}}) = MRFingerprinting.match(D, x, indices, stride, step)
	return function match(x::AbstractVector{<: Number}...)
		x = reshape.(x, num_f, num_σ)
		x = transpose.(x)
		matches, overlap = match(x)
		return matches
	end
end



"""
"""
function admm(
	b::AbstractVector{<: Complex}, # vec([spatial dimensions, singular component])
	A::LinearMap, # PSF
	Ar::LinearMap, # PSF regularised
	P::LinearMap,
	ρ::Real, # Weighting of dictionary regularisation term, in theory the weighting is ρ/2, but here just ρ is used!
	matches::AbstractVector{<: Integer}, # is modified
	match::Function, # Must support match(x) and match(x,y)
	maxiter::Integer, # 48
	cg_maxiter # 64
)
	@assert maxiter > 0
	# Do it specialised for MRF because P is not Vector in the implementation, but mathematically it is
	# See Boyd2010

	# First iteration with P = I, y = 0
	x = cg(A, b, maxiter=cg_maxiter) # How many iterations here?
	matches .= match(x)
	y = x - P*x
	# Allocate space
	br = Vector{ComplexF64}(undef, length(x))
	# All other iterations
	for i = 1:maxiter-1
		# Construct right hand side of normal equations
		br .= b .- ρ .* (y .- P*y)
		# x
		cg!(x, Ar, br, maxiter=cg_maxiter)
		# P
		matches .= match(x, y) # P is updated because it has pointer to `matches`
		# y
		y .+= x .- P * x
		# Stopping criteria TODO: check Asslander's code
	end
	return x, y
end

