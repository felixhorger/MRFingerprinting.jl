
# Definitions: Use flat spatial dimensions, i.e. vector is a matrix with one spatial and one temporal dimension.
# For all the CG stuff however, this needs to be reshaped into a single vector

function sparse2dense(a::AbstractVector{<: Number}, timepoints::Integer, indices::AbstractVector{<: Integer}...) where N
	@assert all(length(a) .== length.(indices))
	b = zeros(eltype(a), timepoints, maximum.(indices)...)
	for (i, j) in enumerate(zip(indices...))
		b[mod1(i, timepoints), j...] = a[i]
	end
	return b
end
# TODO: Function from indices (above) to lr representation, picking out elements from VT/Vconj, simple for-loop will do
# This should then be used also in the function below

function low_rank_mask(mask::AbstractArray{<: Number, N}, V::AbstractMatrix{<: T}, VH::AbstractMatrix{<: T}) where {T, N}
	# mask[time, space]
	# V[time, singular component]
	#=
		For each voxel, mask singular vectors in V,
		then do inner product with vectors in VH to form one matrix per voxel,
		transforming signals from the temporal low-rank domain into itself.
	=#
	shape = size(mask)[2:N]
	lr_mask = Array{T}(undef, size(VH, 1), size(VH, 1), shape...)
	@views for I in CartesianIndices(shape)
		lr_mask[:, :, I] = VH * (mask[:, I] .* V)
	end
	return lr_mask
end


function convenient_Vs(VH::AbstractMatrix{<: Number})
	# Convenience for getting the more practical versions of V: V* and V^T
	transpose(VH), conj.(VH)
end

# Get the corresponding operator
function plan_spatial_ft(x::AbstractArray{<: Number, N}) where N
	# x[spatial dimensions..., channels, singular components]
	# Pay attention that they have to be applied in pairs! Otherwise scaling
	FFT = plan_fft(x, 1:N-2)
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



"""
	plan_sensitivities(sensitivities::Union{AbstractMatrix{<: Number}, AbstractArray{<: Number, 3}}) = S

x[spatial dimensions..., channels, singular components]
sensitivities[spatial dimensions..., channels]

"""
function plan_sensitivities(
	x::AbstractArray{<: Number, N},
	sensitivities::AbstractArray{<: Number, M}
) where {N,M}
	@assert N == M + 1

	# Get dimensions
	# TODO: check spatial dims
	num_σ = size(x, N)
	shape = size(sensitivities)
	spatial_dimensions = prod(shape[1:end-1])
	channels = shape[end]
	input_dimension = spatial_dimensions * num_σ
	output_dimension = input_dimension * channels

	# Reshape
	sensitivities = reshape(sensitivities, spatial_dimensions, channels, 1)
	conj_sensitivities = conj.(sensitivities)

	S = LinearMap{ComplexF64}(
		x::AbstractVector{<: Complex} -> begin
			Sx = sensitivities .* reshape(x, spatial_dimensions, 1, num_σ)
			vec(Sx)
		end,
		y::AbstractVector{<: Complex} -> begin
			y = reshape(y, spatial_dimensions, channels, num_σ)
			SHy = sum(conj_sensitivities .* y; dims=2)
			vec(SHy)
		end,
		output_dimension, input_dimension
	)
	return S
end



"""
	plan_lr2time(x::AbstractArray{<: Number, N}, V_conj::AbstractMatrix{<: Number}, VT::AbstractMatrix{<: Number}) where N

x[spatial dimensions..., channels, singular components]

"""
function plan_lr2time(x::AbstractArray{<: Number, N}, V_conj::AbstractMatrix{<: Number}, VT::AbstractMatrix{<: Number}) where N
	# TODO: This could be done with @turbo
	time, num_σ = size(V_conj)
	spatial_dimensions = prod(size(x)[1:N-1]) # Includes channels if present
	input_dimension = num_σ * spatial_dimensions 
	output_dimension = time * spatial_dimensions
	Λ = LinearMap{ComplexF64}(
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

	function apply_lr_mask(y::AbstractVector{C}, lr_mask::AbstractArray{R, 3}, channels::Integer) where {C <: Complex, R <: Real}

lr_masks[σ, σ, spatial dimensions]

"""
function apply_lr_mask(
	y::AbstractVector{C},
	lr_mask::AbstractArray{<: Real, 3},
	channels::Integer # Makes sense to use Val{N}?
) where C <: Complex
	num_σ = size(lr_mask, 1)
	# TODO: check shape
	y = reshape(y, size(lr_mask, 3), channels, num_σ) # Reshapes allocate ...
	yd = decomplexify(y)
	ymd = similar(yd)
	@turbo for x in axes(lr_mask, 3), c = 1:channels
		for σ2 = 1:num_σ
			ym_real = 0.0
			ym_imag = 0.0
			for σ1 = 1:num_σ
				ym_real += yd[1, x, c, σ1] * lr_mask[σ1, σ2, x]
				ym_imag += yd[2, x, c, σ1] * lr_mask[σ1, σ2, x]
			end
			ymd[1, x, c, σ2] = ym_real
			ymd[2, x, c, σ2] = ym_imag
		end
	end
	ym = reinterpret(C, vec(ymd))
	return ym
end
function apply_lr_mask(
	y::AbstractVector{C},
	lr_mask_d::AbstractArray{<: Real, 4},
	channels::Integer
) where C <: Complex
	num_σ = size(lr_mask, 3)
	y = reshape(y, size(lr_masks, 3), channels, num_σ)
	yd = decomplexify(y)
	ymd = similar(yd) # y *m*asked and *d*ecomplexified
	@turbo for x = axes(lr_mask, 3), c = 1:channels
		for σ2 = 1:num_σ
			ym_real = 0.0
			ym_imag = 0.0
			for σ1 = 1:num_σ
				ym_real += (
					  yd[1, x, c, σ1] * lr_mask_d[1, σ1, σ2, x]
					- yd[2, x, c, σ1] * lr_mask_d[2, σ1, σ2, x]
				)
				ym_imag += (
					  yd[1, x, c, σ1] * lr_mask_d[1, σ1, σ2, x]
					+ yd[2, x, c, σ1] * lr_mask_d[2, σ1, σ2, x]
				)
			end
			ymd[1, x, c, σ2] = ym_real
			ymd[2, x, c, σ2] = ym_imag
		end
	end
	ym = reinterpret(C, vec(ymd))
	return ym
end



"""
	plan_lr_masking(lr_mask::AbstractArray{<: Number, N}, channels::Integer) where N

x[spatial dimensions..., channels, singular components]
sensitivites[spatial dimensions..., channels]
lr_mask[singular components, singular components, spatial dimensions...]

"""
function plan_lr_masking(lr_mask::AbstractArray{<: Number, N}, channels::Integer) where N
	shape = size(lr_mask)
	spatial_shape = shape[3:N]
	num_σ = shape[1]
	@assert shape[2] == num_σ
	spatial_dimensions = prod(spatial_shape)
	lr_mask = reshape(lr_mask, num_σ, num_σ, spatial_dimensions) # It isn't copied, is that good? No...
	lr_mask_d = decomplexify(lr_mask)
	dimension = spatial_dimensions * channels * num_σ
	# Define function
	M = LinearMap{ComplexF64}(
		y::AbstractVector{<: Complex} -> begin
			apply_lr_mask(y, lr_mask_d, channels)
		end,
		dimension,
		ishermitian=true
	)
	return M
end



"""
	plan_lr2lr(F::LinearMap, M::LinearMap [, S::LinearMap])

"""
@inline plan_lr2lr(F::LinearMap, M::LinearMap) = F' * M * F
@inline plan_lr2lr(F::LinearMap, M::LinearMap, S::LinearMap) = S' * F' * M * F * S



"""
	projection_matrix(x::AbstractArray{C, N}, DT_renorm::AbstractMatrix{<: Real}, matches::AbstractVector{<: Integer}) where {C <: Complex, N}

x[spatial dimensions and channels..., singular components]

"""
function projection_matrix(
	x::AbstractArray{C, N}, # TODO: I don't like this concept, I should use type and shape ...
	DT_renorm::AbstractMatrix{<: Real},
	matches::AbstractVector{<: Integer}
) where {C <: Complex, N}
	# `matches` must be preallocated and filled to update P
	# DT_renorm[σ, fingerprints] must be normalised in the SVD domain after cut-off

	shape = size(x)
	spatial_dimensions = prod(shape[1:N-2])
	@assert length(matches) == spatial_dimensions
	num_σ = shape[N]
	@assert size(DT_renorm, 1) == num_σ
	dimension = spatial_dimensions * num_σ
	num_D = size(DT_renorm, 2)

	P = LinearMap{ComplexF64}(
		x::AbstractVector{<: Complex} -> begin
			@assert all(1 .<= matches .<= num_D)
			x = reshape(x, spatial_dimensions, num_σ) # Weird: this has to be done before complexifying, otherwise the same error as in overlap!() with views
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
		dimension,
		ishermitian=true
	)
	return P
end



"""
	plan_lr2lr_regularised(n::Integer, A::LinearMap, P::LinearMap)


"""
function plan_lr2lr_regularised(A::LinearMap, P::LinearMap)
	Ar = LinearMap{ComplexF64}(
		(x -> A*x + x - P*x),
		size(A, 1),
		ishermitian=true
	)
	return Ar
end



"""
"""
function admm(
	b::AbstractVector{<: Complex}, # vec([spatial dimensions, singular component])
	A::LinearMap, # lr2lr
	Ar::LinearMap, # lr2lr regularised
	P::LinearMap,
	matches::AbstractVector{<: Integer},
	find_matches::Function,
	maxiter::Integer
)
	# Do it specialised for MRF because P is not Vector in the implementation, but mathematically it is
	# See Boyd2010
	x = cg(A, b, maxiter=64) # Initial value is computed from standard low-rank reconstruction,
	# TODO: above, how many iterations since noise amplification?
	y = zeros(ComplexF64, length(x))
	br = Vector{ComplexF64}(undef, length(x))
	for i = 1:maxiter
		# Construct right hand side of normal equations
		br .= b .- y .+ P*y
		# x
		x = cg(Ar, br, maxiter=1024)
		# P
		matches .= find_matches(x, y)
		# P is updated because it has pointer to `matches`
		# y
		y .+= x - P*x
		# TODO: Stopping criteria, check Asslander's code
	end
	return x, y
end
# TODO: Get tests from 20220401_Simulation






# Wrong and obsolete?
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
	# Channel and singular component dimension must be flat
	y = fft(x, 1:N-1)
	lr2time(x, VT)
	return yt
end
function kt2lr(yt::AbstractArray{<: Number, N}, V_conj::AbstractMatrix{<: Number}) where N
	# Channel and singular component dimension must be flat
	time2lr(yt, V_conj)
	y = ifft(yt, 1:N-1)
	return y
end
