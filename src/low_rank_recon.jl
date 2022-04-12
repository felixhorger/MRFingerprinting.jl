
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

function lr2time(x::AbstractArray{<: Number, N}, VT::AbstractMatrix{<: Number}) where N
	shape = size(x)
	x = reshape(x, :, shape[N])
	xt = x * VT
	xt = reshape(xt, shape[1:N-1]..., size(VT, 2))
	return xt
end
function time2lr(xt::AbstractArray{<: Number, N}, V_conj::AbstractMatrix{<: Number}) where N
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
	y = reinterpret(real(C), y)
	yt = Matrix{real(C)}(undef, size(y))
	@turbo for x in axes(lr_mask, 1)
		for σ2 = 1:num_σ
			yt_real = 0.0
			yt_imag = 0.0
			for σ1 = 1:num_σ
				yt_real += y[2x-1,	σ1] * lr_mask[x, σ1, σ2]
				yt_imag += y[2x,	σ1] * lr_mask[x, σ1, σ2]
			end
			yt[2x-1,	σ2] = yt_real
			yt[2x,		σ2] = yt_imag
		end
	end
	yt = reinterpret(C, yt)
	yt = vec(yt)
	return yt
end
function apply_lr_mask(y::AbstractVector{C1}, lr_mask::AbstractArray{C2, 3}) where {C1 <: Complex, C2 <: Complex}
	num_σ = size(lr_mask, 3)
	lr_mask = reinterpret(real(C2), lr_mask)
	y = reshape(y, :, num_σ)
	y = reinterpret(real(C1), y)
	yt = Matrix{real(C1)}(undef, size(y))
	@turbo for x = axes(lr_mask, 1) .÷ 2
		for σ2 = 1:num_σ
			yt_real = 0.0
			yt_imag = 0.0
			for σ1 = 1:num_σ
				yt_real += (
					y[2x-1,	σ1] * lr_mask[2x-1,	σ1, σ2]
					- y[2x,	σ1] * lr_mask[2x,	σ1, σ2]
				)
				yt_imag += (
					y[2x-1,	σ1] * lr_mask[2x,	σ1, σ2]
					+ y[2x,	σ1] * lr_mask[2x-1,	σ1, σ2]
				)
			end
			yt[2x-1,	σ2] = yt_real
			yt[2x,		σ2] = yt_imag
		end
	end
	yt = reinterpret(C1, yt)
	yt = vec(yt)
	return yt
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





