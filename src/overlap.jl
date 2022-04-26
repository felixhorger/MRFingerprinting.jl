
# Functions to work with LoopVectorzation and complex arrays
@inline function decomplexify(a::AbstractArray{C}) where C <: Complex
	reinterpret(reshape, real(C), a)
end
@inline decomplexify(a::AbstractArray{<: Real}) = a

@inline function recomplexify(a::AbstractArray{R, N}) where {R <: Real, N}
	@assert N > 1
	reinterpret(reshape, Complex{R}, a)
end



# Real dictionary and real fingerprints
@inline function overlap!(
	y::AbstractMatrix{<: Real},
	A::AbstractMatrix{<: Real},
	x::AbstractMatrix{<: Real}
)
	@turbo for ai in axes(A, 1), xi in axes(x, 2)
		v = 0.0
		for j in axes(A, 2)
			v += A[ai, j] * x[j, xi]
		end
		y[ai, xi] = v^2
	end
	return
end
@inline function overlap!(
	y::AbstractMatrix{<: Real},
	A::AbstractMatrix{<: Real},
	x::NTuple{2, <: AbstractMatrix{<: Real}}
)
	x, z = x
	@turbo for ai in axes(A, 1), xi in axes(x, 2) 
		v = 0.0
		w = 0.0
		for j in axes(A, 2)
			v += A[ai, j] * x[j, xi]
			w += A[ai, j] * z[j, xi]
		end
		y[ai, xi] = v^2 + 2 * v * w
	end
	return
end

# Real dictionary and complex fingerprints
@inline function overlap!(
	y::AbstractMatrix{<: Real},
	A::AbstractMatrix{<: Real},
	x::AbstractArray{<: Real, 3}
)
	@turbo for ai in axes(A, 1), xi in axes(x, 3)
		v_real = 0.0
		v_imag = 0.0
		for j in axes(A, 2)
			v_real += A[ai, j] * x[1, j, xi]
			v_imag += A[ai, j] * x[2, j, xi]
		end
		y[ai, xi] = v_real^2 + v_imag^2
	end
	return
end

@inline function overlap!(
	y::AbstractMatrix{<: Real},
	A::AbstractMatrix{<: Real},
	x::NTuple{2, <: AbstractArray{<: Real, 3}}
	# TODO: @turbo sometimes fails if this is view, see https://github.com/JuliaSIMD/LoopVectorization.jl/issues/365
	# But maybe it requires a strided array. TODO: Adjust type accordingly
)
	# No size assertions done!
	x, z = x
	@turbo for ai in axes(A, 1), xi in axes(x, 3)
		v_real = 0.0
		v_imag = 0.0
		w_real = 0.0
		w_imag = 0.0
		for j in axes(A, 2)
			v_real += A[ai, j] * x[1, j, xi]
			v_imag += A[ai, j] * x[2, j, xi]
			w_real += A[ai, j] * z[1, j, xi]
			w_imag += A[ai, j] * z[2, j, xi]
		end
		y[ai, xi] = (
			v_real^2 + v_imag^2						# ||ax||^2
			+ 2 * (v_real*w_real + v_imag*w_imag)	# 2Re{z^H a * a^H x}
			# where a is a row of A
		)
	end
	return
end



# Complex dictionary and complex fingerprints
@inline function overlap!(
	y::AbstractMatrix{<: Real},
	A::AbstractArray{<: Real, 3}, # Rows of A are conjugated below
	x::AbstractArray{<: Real, 3}
)
	# Mind the axis!
	@turbo for ai in axes(A, 2), xi in axes(x, 3) # Make this loop structure a macro, then make several functions replacing the inner part
		v_real = 0.0
		v_imag = 0.0
		for j in axes(A, 3)
			v_real += A[1, ai, j] * x[1, j, xi] + A[2, ai, j] * x[2, j, xi]
			v_imag += A[1, ai, j] * x[2, j, xi] - A[2, ai, j] * x[1, j, xi]
		end
		y[ai, xi] = v_real^2 + v_imag^2
	end
	return
end

@inline function overlap!(
	y::AbstractMatrix{<: Real},
	A::AbstractArray{<: Real, 3},
	x::NTuple{2, <: AbstractArray{<: Real, 3}}
	# TODO: @turbo sometimes fails if this is view, see https://github.com/JuliaSIMD/LoopVectorization.jl/issues/365
	# But maybe it requires a strided array. TODO: Adjust type accordingly
)
	# No size assertions done!
	x, z = x
	@turbo for ai in axes(A, 2), xi in axes(x, 3)
		v_real = 0.0
		v_imag = 0.0
		w_real = 0.0
		w_imag = 0.0
		for j in axes(A, 3)
			v_real += A[1, ai, j] * x[1, j, xi] + A[2, ai, j] * x[2, j, xi]
			v_imag += A[1, ai, j] * x[2, j, xi] - A[2, ai, j] * x[1, j, xi]
			w_real += A[1, ai, j] * z[1, j, xi] + A[2, ai, j] * z[2, j, xi]
			w_imag += A[1, ai, j] * z[2, j, xi] - A[2, ai, j] * z[1, j, xi]
		end
		y[ai, xi] = (
			v_real^2 + v_imag^2						# ||a^H x||^2
			+ 2 * (v_real*w_real + v_imag*w_imag)	# 2Re{z^H a * a^H x} where a is a row of A
		)
	end
	return
end

