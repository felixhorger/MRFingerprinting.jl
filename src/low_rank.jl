
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
	indices::AbstractVector{<: NTuple{N, Integer}}
) where N
	@assert all(size(x, 3) .== length.(indices))
	@assert all(all(0 .< indices[d] .<= shape[d]) for d ∈ 1:N)
	timepoints = size(VH, 2)
	y = zeros(ComplexF64, size(x, 1), shape..., size(x, 2), size(VH, 1)) # spatial dims..., channels, sigma
	# Must be zeros because if a location isn't sampled it would contain undefs
	@inbounds for (i, j) in enumerate(indices)
		t = mod1(i, timepoints)
		for σ in axes(VH, 1)
			y[:, j..., :, σ] += VH[σ, t] * x[:, :, i]
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
	mask[time, space]
	VH[singular component, time]
	
	For each voxel, mask singular vectors in V,
	then do inner product with vectors in VH to form one matrix per voxel,
	transforming signals from the temporal low-rank domain into itself.
	
	Note: There is no use case of difference phase encoding steps with overlapping readouts (k-wise) I know of,
	so this assumes only the same phase encoding step overlaps with itself (to compute the mask).
"""
function low_rank_mask(
	V_conj::AbstractMatrix{<: T},
	VT::AbstractMatrix{<: T},
	indices::AbstractVector{<: NTuple{N, Integer}},
	shape::NTuple{N, Integer},
	num_dynamic::Integer
) where {N, T}
	num_readouts = prod(shape)
	num_σ = size(VT, 1)
	linear_indices = LinearIndices(shape)
	perm = sortperm(indices; by=(t::NTuple{N, Integer} -> linear_indices[t...]))
	lr_mask = zeros(T, size(VT, 1), size(VT, 1), shape...)
	for i in eachindex(perm)
		j = perm[i]
		x = indices[j]
		dynamic = mod1(j, num_dynamic)
		mixing_matrix = @view lr_mask[:, :, x...]
		@turbo for σ1 = 1:num_σ, σ2 = 1:num_σ
			mixing_matrix[σ1, σ2] += VT[σ1, dynamic] * V_conj[dynamic, σ2] # Outer product
		end
	end
	return lr_mask
end

"""

lr_masks[σ, σ, spatial dimensions]

"""
function apply_lr_mask(
	y::AbstractVector{C},
	lr_mask::AbstractArray{<: Real, 3}, # Flat spatial dimension
	readout_length::Integer, # Needs to be separate because of the order in which y usually is
	channels::Integer # Makes sense to use Val{N}?
) where C <: Complex
	num_σ = size(lr_mask, 1)
	# TODO: check shape
	y = reshape(y, readout_length, size(lr_mask, 3), channels, num_σ) # Reshapes allocate ...
	yd = decomplexify(y)
	ymd = similar(yd)
	@turbo for x in axes(lr_mask, 3), c = 1:channels, s = 1:readout_length
		for σ2 = 1:num_σ
			ym_real = 0.0
			ym_imag = 0.0
			for σ1 = 1:num_σ
				ym_real += yd[1, s, x, c, σ1] * lr_mask[σ1, σ2, x]
				ym_imag += yd[2, s, x, c, σ1] * lr_mask[σ1, σ2, x]
			end
			ymd[1, s, x, c, σ2] = ym_real
			ymd[2, s, x, c, σ2] = ym_imag
		end
	end
	ym = reinterpret(C, vec(ymd))
	return ym
end
function apply_lr_mask(
	y::AbstractVector{C},
	lr_mask_d::AbstractArray{<: Real, 4},
	readout_length::Integer, # Needs to be separate because of the order in which y usually is
	channels::Integer
) where C <: Complex
	num_σ = size(lr_mask, 3)
	y = reshape(y, readout_length, size(lr_masks, 3), channels, num_σ)
	yd = decomplexify(y)
	ymd = similar(yd) # y *m*asked and *d*ecomplexified
	@turbo for x = axes(lr_mask, 3), c = 1:channels, s = 1:readout_length
		for σ2 = 1:num_σ
			ym_real = 0.0
			ym_imag = 0.0
			for σ1 = 1:num_σ
				ym_real += (
					  yd[1, s, x, c, σ1] * lr_mask_d[1, σ1, σ2, x]
					- yd[2, s, x, c, σ1] * lr_mask_d[2, σ1, σ2, x]
				)
				ym_imag += (
					  yd[1, s, x, c, σ1] * lr_mask_d[1, σ1, σ2, x]
					+ yd[2, s, x, c, σ1] * lr_mask_d[2, σ1, σ2, x]
				)
			end
			ymd[1, s, x, c, σ2] = ym_real
			ymd[2, s, x, c, σ2] = ym_imag
		end
	end
	ym = reinterpret(C, vec(ymd))
	return ym
end



"""

lr_mask[σ1, σ2, spatial dimensions...]
lr_mask not copied
"""
function plan_lr_masking(lr_mask::AbstractArray{<: Number, N}, readout_length::Integer, num_channels::Integer) where N
	# Get and check shapes
	lr_mask_shape = size(lr_mask)
	num_σ = lr_mask_shape[1]
	@assert lr_mask_shape[2] == num_σ
	shape = lr_mask_shape[3:N]
	num_phase_encode = prod(shape)
	# Reshape and split real/imag
	lr_mask = reshape(lr_mask, num_σ, num_σ, num_phase_encode) # It isn't copied
	lr_mask_d = decomplexify(lr_mask)
	# Define function
	M = LinearMap{ComplexF64}(
		y::AbstractVector{<: Complex} -> begin
			apply_lr_mask(y, lr_mask_d, readout_length, num_channels)
		end,
		num_phase_encode * readout_length * num_channels * num_σ,
		ishermitian=true
	)
	return M
end
"""
shape includes channels
"""
function plan_lr_masking(
	V_conj::AbstractMatrix{<: Number},
	VT::AbstractMatrix{<: Number},
	indices::AbstractVector{<: NTuple{N, Integer}},
	shape::NTuple{N, Integer},
	readout_length::Integer,
	num_channels::Integer,
	num_dynamic::Integer
) where N
	lr_mask = low_rank_mask(V_conj, VT, indices, shape, num_dynamic)
	return plan_lr_masking(lr_mask, readout_length, num_channels)
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
	Note that M can be a low-rank mask, i.e. this acts on a vector
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
)::NTuple{2, Vector{ComplexF64}}
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

