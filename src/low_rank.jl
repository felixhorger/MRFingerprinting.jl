
# Definitions: Use flat spatial dimensions, i.e. vector is a matrix with one spatial and one temporal dimension.
# For all the CG stuff however, this needs to be reshaped into a single vector

function sparse2dense(a::AbstractVector{<: Number}, timepoints::Integer, indices::AbstractVector{<: Integer}...) where N
	@assert all(length(a) .== length.(indices))
	b = zeros(eltype(a), maximum.(indices)..., timepoints)
	for (i, j) in enumerate(zip(indices...))
		b[j..., mod1(i, timepoints)] = a[i]
	end
	return b
end

function low_rank_mask(mask::AbstractArray{<: Number, N}, V::AbstractMatrix{<: T}, VH::AbstractMatrix{<: T}) where {T, N}
	# mask[space, time]
	# V[time, singular component]
	#=
		For each voxel, mask singular vectors in V,
		then do inner product with vectors in VH to form one matrix per voxel,
		transforming signals from the temporal low-rank domain into itself.
	=#
	shape = size(mask)[1:N-1]
	lr_mask = Array{T}(undef, shape..., size(VH, 1), size(VH, 1))
	@views for I in CartesianIndices(shape)
		lr_mask[I, :, :] = VH * (mask[I, :] .* V)
	end
	return lr_mask
end


function convenient_Vs(VH::AbstractMatrix{<: Number})
	# Convenience for getting the more practical versions of V: V* and V^T
	transpose(VH), conj.(VH)
end

function spatial_vol2vec(x::AbstractArray{<: Number, N}) where N
	# x[space..., time]
	reshape(x, :, size(x, N))
end
function spatial_vec2vol(x::AbstractMatrix{<: Number}, shape::NTuple{N, Int64}) where N
	# x[space, time]
	# shape is spatial dimensions
	reshape(x, shape..., size(x, 2))
end

# Get the corresponding operator
function plan_spatial_ft(x::AbstractArray{<: Number, N}) where N
	# Pay attention that they have to be applied in pairs! Otherwise scaling
	FFT = plan_fft(x, 1:N-1)
	FFTH = inv(FFT)
	shape = size(x)
	restore_shape(x) = reshape(x, shape) # Note that x is Vector
	F = LinearMap{ComplexF64}(
		x -> begin
			x = restore_shape(x)
			y = FFT * x
			vec(y)
		end,
		y -> let
			y = restore_shape(y)
			x = FFTH * y
			vec(x)
		end,
		prod(shape)
	)
	return F
end

# TODO: Need non-allocating version, one memory block for each representation? check cg()
function lr2time(x::AbstractArray{<: Number, N}, VT::AbstractMatrix{<: Number}) where N
	# TODO: This could be done with @turbo
	shape = size(x)
	x = reshape(x, :, shape[N])
	xt = x * VT
	xt = reshape(xt, shape[1:N-1]..., size(VT, 2))
	return xt
end
function time2lr(xt::AbstractArray{<: Number, N}, V_conj::AbstractMatrix{<: Number}) where N
	# TODO: This could be done with @turbo
	shape = size(xt)
	xt = reshape(xt, :, shape[N])
	x = xt * V_conj
	x = reshape(x, shape[1:N-1]..., size(V_conj, 2))
	return x
end
function lr2kt(x::AbstractArray{<: Number, N}, VT::AbstractMatrix{<: Number}) where N
	y = fft(x, 1:N-1)
	lr2time(x, VT)
	return yt
end
function kt2lr(yt::AbstractArray{<: Number, N}, V_conj::AbstractMatrix{<: Number}) where N
	time2lr(yt, V_conj)
	y = ifft(yt, 1:N-1)
	return y
end

function plan_lr2time(x::AbstractArray{<: Number, N}, V_conj::AbstractMatrix{<: Number}, VT::AbstractMatrix{<: Number}) where N
	# TODO: This could be done with @turbo
	time, num_σ = size(V_conj)
	spatial_dimensions = prod(size(x)[1:N-1])
	input_dimension = num_σ * spatial_dimensions 
	output_dimension = time * spatial_dimensions
	L = LinearMap{ComplexF64}(
		y -> begin
			y = reshape(y, :, num_σ)
			yt = y * VT
			vec(yt)
		end,
		yt -> begin
			yt = reshape(yt, :, time)
			y = yt * V_conj
			vec(y)
		end,
		output_dimension,
		input_dimension
	)
	return L
end

function plan_lr2kt(
	x::AbstractArray{<: Number, N},
	V_conj::AbstractMatrix{<: Number},
	VT::AbstractMatrix{<: Number},
) where N
	# TODO: might make sense to preallocate an intermediate state?
	F = plan_spatial_ft(x)
	L = plan_lr2time(x, V_conj, VT)
	return L * F
end

function apply_lr_mask(y::AbstractVector{C}, lr_mask::AbstractArray{R, 3}) where {C <: Complex, R <: Real}
	num_σ = size(lr_mask, 3)
	y = reshape(y, :, num_σ) # Reshapes allocate ...
	yd = decomplexify(y)
	ymd = similar(yd)
	@turbo for x in axes(lr_mask, 1)
		for σ2 = 1:num_σ
			ym_real = 0.0
			ym_imag = 0.0
			for σ1 = 1:num_σ
				ym_real += yd[1, x, σ1] * lr_mask[x, σ1, σ2]
				ym_imag += yd[2, x, σ1] * lr_mask[x, σ1, σ2]
			end
			ymd[1, x, σ2] = ym_real
			ymd[2, x, σ2] = ym_imag
		end
	end
	ym = reinterpret(C, vec(ymd))
	return ym
end
function apply_lr_mask(y::AbstractVector{C1}, lr_mask::AbstractArray{C2, 3}) where {C1 <: Complex, C2 <: Complex}
	num_σ = size(lr_mask, 3)
	lr_mask_d = decomplexify(lr_mask)
	y = reshape(y, :, num_σ)
	yd = decomplexify(y)
	ymd = similar(yd) # y *m*asked and *d*ecomplexified
	@turbo for x = axes(lr_mask, 1)
		for σ2 = 1:num_σ
			ym_real = 0.0
			ym_imag = 0.0
			for σ1 = 1:num_σ
				ym_real += (
					  yd[1, x, σ1] * lr_mask_d[1, x, σ1, σ2]
					- yd[2, x, σ1] * lr_mask_d[2, x, σ1, σ2]
				)
				ym_imag += (
					  yd[1, x, σ1] * lr_mask_d[1, x, σ1, σ2]
					+ yd[2, x, σ1] * lr_mask_d[2, x, σ1, σ2]
				)
			end
			ymd[1, x, σ2] = ym_real
			ymd[2, x, σ2] = ym_imag
		end
	end
	ym = reinterpret(C1, vec(ymd))
	return ym
end
function plan_lr_masking(lr_mask::AbstractArray{<: Number, N}) where N
	shape = size(lr_mask)
	spatial_shape = shape[1:N-2]
	num_σ = shape[N]
	lr_mask = reshape(lr_mask, prod(spatial_shape), num_σ, num_σ)
	# Define function
	M = LinearMap{ComplexF64}(
		y -> apply_lr_mask(y, lr_mask),
		prod(spatial_shape) * num_σ,
		ishermitian=true
	)
	return M
end
function plan_lr2lr(
	x::AbstractArray{<: Number, N},
	lr_mask::AbstractArray{<: Number, K}
) where {N, K}
	@assert N == K-1
	shape = size(lr_mask)[1:N]
	num_σ = size(lr_mask)[K]
	@assert shape == size(x)
	@assert shape[N] == num_σ
	M = plan_lr_masking(lr_mask)
	F = plan_spatial_ft(x)
	return F' * M * F
end
# TODO: add functions to generate this from F and M


function projection_matrix(
	x::AbstractArray{C, N},
	DT_renorm::AbstractMatrix{<: Real},
	matches::AbstractVector{<: Integer}
) where {C <: Complex, N}
	# `matches` must be preallocated and filled to update P
	# DT_renorm[σ, fingerprints] must be normalised in the SVD domain after cut-off

	shape = size(x)
	spatial_dimensions = prod(shape[1:N-1])
	num_σ = shape[N]
	n = spatial_dimensions * num_σ
	num_D = size(DT_renorm, 2)
	@assert length(matches) == spatial_dimensions
	@assert size(DT_renorm, 1) == num_σ

	P = LinearMap{ComplexF64}(
		x::AbstractVector{<: Complex} -> begin
			@assert all(1 .<= matches .<= num_D)
			xd = reinterpret(real(C), x) # d for decomplexified
			xd = reshape(xd, 2, spatial_dimensions, num_σ)
			xpd = similar(xd)
			for xi in eachindex(matches)
				@inbounds match = matches[xi]
				p_real = 0.0
				p_imag = 0.0
				@turbo for σ = 1:num_σ # Shame, depends on execution order
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
		n,
		ishermitian=true
	)
	return P
end

function plan_lr2lr_regularised(x::AbstractArray{<: Complex}, A::LinearMap, P::LinearMap)
	B = LinearMap{ComplexF64}(
		(x -> A*x + x - P*x),
		prod(size(x)),
		ishermitian=true
	)
	return B
end

function admm(
	s::AbstractVector{<: Complex}, # vec([spatial dimensions, time])
	L::LinearMap, # lr2kt
	A::LinearMap, # lr2lr
	Ar::LinearMap, # lr2lr regularised
	P::LinearMap,
	matches::AbstractVector{<: Integer},
	find_matches::Function,
	maxiter::Integer
)
	# Do it specialised for MRF because P is not Vector in the implementation, but mathematically it is
	# See Boyd2010
	backprojection = L' * s
	x = cg(A, backprojection, maxiter=64) # Initial value is computed from standard low-rank reconstruction,
	# TODO: above, how many iterations since noise amplification?
	y = zeros(ComplexF64, length(x))
	b = Vector{ComplexF64}(undef, length(x))
	for i = 1:maxiter
		# Construct right hand side of normal equations
		b .= backprojection .- y .+ P*y
		# x
		x = cg(Ar, b, maxiter=1024)
		# P
		matches .= find_matches(x, y) # This takes long (overlap!()), need AVX or GPU
		# P is updated because it has pointer to `matches`
		# TODO asymmetric overlap remember Dx to use below
		# y
		y .+= x - P*x
		# TODO: Stopping criteria, check Asslander's code
	end
	return x, y
end

# TODO: Get tests from 20220401_Simulation

