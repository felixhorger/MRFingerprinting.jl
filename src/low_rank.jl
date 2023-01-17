
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
function plan_lr2time(
	V_conj::AbstractMatrix{<: Number},
	VT::AbstractMatrix{<: Number},
	num_other::Integer;
	dtype::Type{T}=ComplexF64
) where T <: Number
	# TODO: This could be done with @turbo
	num_time, num_σ = size(V_conj)
	@assert size(VT) == (num_σ, num_time)
	Λ = LinearOperator{T}(
		(num_time * num_other, num_σ * num_other),
		(yt, y) -> begin
			mul!(reshape(yt, num_other, num_time), reshape(y, num_other, num_σ), VT)
			yt
		end;
		adj = (x, xt) -> begin
			mul!(reshape(x, num_other, num_σ), reshape(xt, num_other, num_time), V_conj)
			x
		end,
		out=Vector{T}(undef, num_other * num_time),
		out_adj_inv=Vector{T}(undef, num_other * num_σ)
	)
	return Λ
end




"""
	plan_lr2kt(L::AbstractLinearOperator, F::AbstractLinearOperator)

"""
@inline plan_lr2kt(Λ::AbstractLinearOperator, F::AbstractLinearOperator) = Λ * F
"""
	plan_lr2kt(L::AbstractLinearOperator, F::AbstractLinearOperator, S::AbstractLinearOperator)

"""
@inline plan_lr2kt(Λ::AbstractLinearOperator, F::AbstractLinearOperator, S::AbstractLinearOperator) = plan_lr2kt(Λ, F) * S



function lowrank_sparse2dense(
	kspace::AbstractArray{<: T, 3}, # [readout, channel, phase encoding]
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer},
	VH::AbstractMatrix{<: Number}
) where {N, T <: Number}
	@assert length(indices) == size(kspace, 3)
	num_σ, num_dynamic = size(VH)
	linear_indices = LinearIndices(shape)
	perm = sortperm(indices; by=(k::CartesianIndex{N} -> linear_indices[k]))
	backprojection = zeros(T, size(kspace, 1), size(kspace, 2), shape..., num_σ)
	for i in eachindex(perm)
		j = perm[i]
		dynamic = mod1(j, num_dynamic)
		k = indices[j]
		for σ = 1:num_σ
			backprojection[:, :, k, σ] += (@views @inbounds kspace[:, :, j] * VH[σ, dynamic])
			# need to check if k is in bounds
		end
	end
	return backprojection
end



"""
	shape is the shape of the phase encoding plane
	VH[singular component, time]
	
	For each voxel, mask singular vectors in V,
	then do inner product with vectors in VH to form one matrix per voxel,
	transforming signals from the temporal low-rank domain into itself.

	indices must be unique in the k-t domain otherwise this will be wrong!
"""
function lowrank_mixing(
	VH::AbstractMatrix{<: T},
	indices::AbstractVector{<: CartesianIndex{N}},
	shape::NTuple{N, Integer}
) where {N, T}
	num_σ, num_dynamic = size(VH)
	linear_indices = LinearIndices(shape)
	perm = sortperm(indices; by=(x::CartesianIndex{N} -> linear_indices[x]))
	lr_mix = zeros(T, num_σ, num_σ, shape...)
	# Precompute outer product of singular vectors (t ≡ dynamic)
	# M_{σ,σ'}	= ∑_{t,t'} V^H_{σ,t} ⋅ U_{t,t'} ⋅ V_{t',σ'}
	#			= ∑_{t,t'} U_{t,t'} ⋅ V_{t',σ'} ⋅ V^*_{t,σ}
	#			= ∑_{t,t'} u_t ⋅ δ_{t,t'} ⋅ V_{t',σ'} ⋅ V^*_{t,σ}
	#			= ∑_t u_t ⋅ V^*_{t,σ} ⋅ V_{t,σ'}
	VH_outer_V = Array{T, 3}(undef, num_σ, num_σ, num_dynamic)
	for dynamic = 1:num_dynamic, σ2 = 1:num_σ, σ1 = 1:num_σ
		VH_outer_V[σ1, σ2, dynamic] = VH[σ1, dynamic] * conj(VH[σ2, dynamic])
	end
	for i in eachindex(perm)
		j = perm[i]
		dynamic = mod1(j, num_dynamic)
		k = indices[j]
		@views lr_mix[:, :, k] .+= VH_outer_V[:, :, dynamic]
	end
	return permutedims(lr_mix, ((3:N+2)..., 1, 2))
end

"""

lr_mix[spatial dimensions, σ, σ]

"""
function apply_lowrank_mixing!(
	ym::AbstractArray{C, 4},
	y::AbstractArray{C, 4},
	lr_mix::AbstractArray{<: Real, N}, # (potentially complex axis) spatial, σ, σ
	ι::Integer,
	κ::Integer
) where {C <: Complex, N}
	@assert pointer(ym) !== pointer(y)
	@assert size(ym) == size(y)
	@assert size(ym, 4) == size(lr_mix, N)
	(ymd, yd) = decomplexify.((ym, y))
	apply_lowrank_mixing!(ymd, yd, lr_mix, ι, κ)
	return ym
end
function apply_lowrank_mixing!(
	ymd::AbstractArray{R, 5},
	yd::AbstractArray{R, 5},
	lr_mix::AbstractArray{<: Real, 3}, # spatial, σ, σ
	ι::Integer,
	κ::Integer
) where R <: Real
	num_spatial = size(lr_mix, 1)
	num_σ = size(lr_mix, 3)
	@tturbo for i in 1:length(ymd) # Cannot use eachindex because ymd is ReinterpretArray, not nice, my opinion
		ymd[i] = 0
	end
	for σ2 = 1:num_σ, σ1 = 1:num_σ
		Threads.@threads for k = 1:κ # TODO: This doesn't seem optimal, yet experimentally gives best performance
			@turbo for i = 1:num_spatial
				l = lr_mix[i, σ1, σ2]
				for j = 1:ι
					ymd[1, j, i, k, σ2] += yd[1, j, i, k, σ1] * l
					ymd[2, j, i, k, σ2] += yd[2, j, i, k, σ1] * l
				end
			end
		end
	end
	return ymd
end
function apply_lowrank_mixing!(
	ymd::AbstractArray{R, 5},
	yd::AbstractArray{R, 5},
	lr_mix_d::AbstractArray{<: Real, 4}, # complex, spatial, σ, σ
	ι::Integer,
	κ::Integer
) where R <: Real
	num_spatial = size(lr_mix_d, 2)
	num_σ = size(lr_mix_d, 3)
	@tturbo for i in 1:length(ymd) # Cannot use eachindex because ymd is ReinterpretArray, not nice, my opinion
		ymd[i] = 0
	end
	for σ2 = 1:num_σ, σ1 = 1:num_σ
		Threads.@threads for k = 1:κ
			@turbo for i = 1:num_spatial
				l_re = lr_mix_d[1, i, σ1, σ2]
				l_im = lr_mix_d[2, i, σ1, σ2]
				for j = 1:ι
					ymd[1, j, i, k, σ2] += (
						  yd[1, j, i, k, σ1] * l_re
						- yd[2, j, i, k, σ1] * l_im
					)
					ymd[2, j, i, k, σ2] += (
						  yd[1, j, i, k, σ1] * l_im
						+ yd[2, j, i, k, σ1] * l_re
					)
				end
			end
		end
	end
	return ymd
end



"""

lr_mix[spatial dimensions..., σ1, σ2]
lr_mix not copied

ι,κ = number of elements in the first,last spatial dimension of y
which is assumed to be acquires at the same time.
For example: 3D Cartesian imaging would have dimensions (num_lines, num_partitions, num_readouts)
where readouts can be Fourier transformed beforehand. In this case ι = 1 and κ = num_readouts.
For stack of stars with fully sampled partition direction, (num_columns, num_lines, num_partitions)
is more favourable, so ι = num_columns and κ = num_partitions.
For multi-channel data, the number of channels and κ can be fused.

"""
function plan_lowrank_mixing(lr_mix::AbstractArray{<: Number, N}, ι::Integer, κ::Integer; dtype::Type{C}=ComplexF64) where {N, C <: Number}
	# Get and check shapes
	lr_mix_shape = size(lr_mix)
	num_σ = lr_mix_shape[N]
	@assert lr_mix_shape[N-1] == num_σ
	shape = lr_mix_shape[1:N-2]
	num_phase_encode = prod(shape)
	# Reshape and split real/imag
	lr_mix_d = decomplexify(reshape(lr_mix, num_phase_encode, num_σ, num_σ)) # If lr_mix is real, this does nothing
	# Define function
	M = HermitianOperator{C}(
		num_phase_encode * ι * κ * num_σ,
		(ym, y) -> begin
			apply_lowrank_mixing!(
				reshape(ym, ι, num_phase_encode, κ, num_σ),
				reshape(y, ι, num_phase_encode, κ, num_σ),
				lr_mix_d,
				ι, κ
			)
			ym
		end;
		out=Vector{C}(undef, ι * num_phase_encode * κ * num_σ)
	)
	return M
end
"""
Convenience
"""
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


"""
	shape = spatial target shape
	not side-effect free
	mutates x
"""
function prepare_lowrank_toeplitz_embedding(F_double_fov::LinearOperator{T}, lr_mix::AbstractArray{<: Number, 3}, shape::NTuple{D, Integer}, num_other::Integer; kwargs...) where {T <: Complex, D}
	double_shape = 2 .* shape
	num_double_fov = size(F_double_fov, 2)
	@assert prod(double_shape) == num_double_fov
	num_σ = size(lr_mix, 2)
	@assert num_σ == size(lr_mix, 3)
	num_phase_encode = size(lr_mix, 1)
	num_columns, remainder_columns = divrem(size(F_double_fov, 1), num_phase_encode)
	@assert remainder_columns == 0

	m = Array{T, 3}(undef, size(F_double_fov, 2), num_σ, num_σ)
	let tmp = Matrix{T}(undef, num_columns, num_phase_encode), tmp_vec = vec(tmp)
		# Compute one half including diagonal
		for σ1 = 1:num_σ, σ2 = 1:σ1
			# Repeat array along readout direction
			for p = 1:num_phase_encode
				@views tmp[:, p] .= lr_mix[p, σ2, σ1]
			end
			# Apply adjoint
			@views m[:, σ2, σ1] .= F_double_fov' * tmp_vec
		end
		# Copy other half (without diagonal)
		for σ1 = 1:num_σ, σ2 = σ1+1:num_σ
			@views m[:, σ2, σ1] .= m[:, σ1, σ2]
		end
	end

	# Precompute kernel, i.e. mixing matrices in new, Cartesian k-space
	Fm = fft(reshape(m, double_shape..., num_σ, num_σ), 1:D) ./ prod(shape) # include fft normalisation
	Fm_d = decomplexify(reshape(Fm, num_double_fov, num_σ, num_σ))

	# Array for padding input vector and array for intermediate step
	x_padded = Array{T, D+2}(undef, double_shape..., num_other, num_σ)
	y_padded = similar(x_padded)

	# Plan Fourier and low-rank mixing
	F = plan_fft!(x_padded, 1:D; kwargs...)
	FH_unnormalised = plan_bfft!(x_padded, 1:D; kwargs...) # Normalisation in Fm
	return x_padded, y_padded, Fm_d, F, FH_unnormalised
end

function apply_lowrank_toeplitz_embedding!(
	y::AbstractArray{T, N},
	x::AbstractArray{T, N},
	x_padded::AbstractArray{T, N},
	y_padded::AbstractArray{T, N},
	Fm_d::AbstractArray{<: Real, K},
	F::FFTW.cFFTWPlan,
	FH_unnormalised::FFTW.cFFTWPlan
) where {T <: Complex, N, K}
	# Dimensions and checks
	shape = size(x)[1:N-2] # First axis is length 2, decomplexified
	double_shape = size(x_padded)[1:N-2]
	num_other, num_σ = size(x)[N-1:N]
	@assert (num_other, num_σ) == size(x_padded)[N-1:N]
	@assert K ∈ (3, 4)
	# Reshape and decomplexify arrays
	(x_d, y_d) = decomplexify.((x, y))
	(x_padded_d, y_padded_d) = decomplexify.((x_padded, y_padded))
	(x_padded_flat, y_padded_flat) = reshape.(
		(x_padded, y_padded),
		1, prod(double_shape), num_other, num_σ
	)
	(x_padded_flat_d, y_padded_flat_d) = decomplexify.((x_padded_flat, y_padded_flat))
	# Zero fill padded array
	@tturbo for i = 1:length(x_padded_d)
		x_padded_d[i] = 0
	end
	# Pad x into x_padded
	zero_offset = Tuple(0 for _ = 1:N+1)
	offset = (0, MRIRecon.centre_offset.(shape)..., 0, 0)
	shape_d = (2, shape..., num_other, num_σ)
	MRIRecon.turbo_block_copyto!(x_padded_d, x_d, shape_d, offset, zero_offset)
	# Fourier transform x_padded
	F * x_padded # In-place
	# Apply low-rank mixing
	apply_lowrank_mixing!(y_padded_flat_d, x_padded_flat_d, Fm_d, 1, num_other)
	# Fourier transform back
	FH_unnormalised * y_padded # In-place
	# Crop into y
	MRIRecon.turbo_block_copyto!(y_d, y_padded_d, shape_d, zero_offset, offset)
	return y
end

"""
F_double_fov includes all non-Cartesian dimensions
"""
function plan_lowrank_toeplitz_embedding(y::AbstractArray{T, N}, F_double_fov::LinearOperator{T}, lr_mix::AbstractArray{<: Number, 3}; kwargs...) where {T <: Complex, N}
	@assert N > 2
	num_other, num_σ = size(y)[N-1:N]
	shape = size(y)[1:N-2]
	@assert num_σ == size(lr_mix, 2) == size(lr_mix, 3)
	x_padded, y_padded, Fm_d, F, FH_unnormalised = prepare_lowrank_toeplitz_embedding(
		F_double_fov,
		lr_mix,
		shape, num_other
	)
	# Define convolution operator
	FHMF = HermitianOperator{T}(
		length(y),
		(y, x) -> begin
			(ys, xs) = reshape.((y, x), shape..., num_other, num_σ)
			apply_lowrank_toeplitz_embedding!(ys, xs, x_padded, y_padded, Fm_d, F, FH_unnormalised)
			y
		end;
		out=vec(y)
	)
	return FHMF
end

"""
In place, be careful with conjugate gradient
"""
function plan_lowrank_toeplitz_embedding!(shape::NTuple{D, Integer}, num_other::Integer, F_double_fov::LinearOperator{T}, lr_mix::AbstractArray{<: Number, 3}; kwargs...) where {T <: Number, D}
	num_σ = size(lr_mix, 2)
	@assert num_σ == size(lr_mix, 3)
	x_padded, y_padded, Fm_d, F, FH_unnormalised = prepare_lowrank_toeplitz_embedding(
		F_double_fov,
		lr_mix,
		shape, num_other
	)
	# Define convolution operator
	FHMF = HermitianOperator{T}(
		prod(shape) * num_other * num_σ,
		(y, x) -> begin
			@assert length(y) == 0
			z = reshape(x, shape..., num_other, num_σ)
			apply_lowrank_toeplitz_embedding!(z, z, x_padded, y_padded, Fm_d, F, FH_unnormalised)
			x
		end
	)
	return FHMF
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

	P = HermitianOperator{C}(
		num_x * num_σ,
		x -> begin
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
			xp = reinterpret(C, xpd)
		end
	)
	return P
end



"""
	plan_psf_regularised(A::HermitianOperator, P::HermitianOperator)
	A = S' * F' * M * F * S, i.e. the PSF without the projection.
	Note that M can be a low-rank mixing, i.e. this acts on a vector
	in temporal low-rank image space.

"""
function plan_psf_regularised(A::HermitianOperator{T}, P::HermitianOperator{T}, ρ::Real) where T <: Number
	Ar = HermitianOperator{T}(
		size(A, 1),
		x -> A*x + ρ * (x - P*x)
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
	b::AbstractVector{T}, # vec([spatial dimensions, singular component])
	A::HermitianOperator{T}, # PSF
	Ar::HermitianOperator{T}, # PSF regularised
	P::HermitianOperator{T},
	ρ::Real, # Weighting of dictionary regularisation term, in theory the weighting is ρ/2, but here just ρ is used!
	matches::AbstractVector{<: Integer}, # is modified
	match::FunctionWrapper{Tuple{Vector{Int64}, Vector{Float64}}, Tuple{Vector{T}}}, # Must support match(x) and match(x,y)
	maxiter::Integer, # 48
	cg_maxiter # 64
) where T <: Complex
	@assert maxiter > 0
	# Do it specialised for MRF because P is not Vector in the implementation, but mathematically it is
	# See Boyd2010

	# First iteration with P = I, y = 0
	error("use cg!() and init backproj? Check Asslaender paper")
	x = cg(A, b, maxiter=cg_maxiter) # How many iterations here?
	matches .= match(x)
	y = x - P*x
	# Allocate space
	br = Vector{T}(undef, length(x))
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

