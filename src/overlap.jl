

# Real
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
	x::T,
	z::T
) where T <: AbstractMatrix{<: Real}

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

# Complex
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
	x::T,
	z::T
) where T <: AbstractArray{<: Real, 3}

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
			+ 2 * (v_real*w_real + v_imag*w_imag2)	# 2Re{z^H a * a^H x}
			# where a is a row of A
		)
	end
	return
end

