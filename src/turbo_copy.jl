
"""
D can be smaller than N
"""
#function block_copyto!(
#	dest::AbstractArray{T, N},
#	src::AbstractArray{T, N},
#	idx::NTuple{D, UnitRange{Int64}},
#	offset::NTuple{D, Int64}
#) where {T <: Number, N, D}
#	other_shape = size(dest)[D+1:N]
#	@assert other_shape == size(src)[D+1:N]
#	for K in CartesianIndices(other_shape)
#		for I in CartesianIndices(idx)
#			J = CartesianIndex(Tuple(I) .- offset)
#			dest[I, K] = src[J, K]
#		end
#	end
#	return dest
#end

@generated function turbo_block_copyto!(
	dest::AbstractArray{T, N},
	src::AbstractArray{T, N},
	shape::NTuple{N, Int64},
	offset_dest::NTuple{N, Int64},
	offset_src::NTuple{N, Int64}
) where {T <: Real, N}
	loops = quote
		@nloops(
			$N, i,
			d -> 1:i_max_d,
			d -> begin
				j_dest_d = i_d + k_dest_d
				j_src_d = i_d + k_src_d
			end,
			begin
				(@nref $N dest j_dest) = @nref $N src j_src
			end
		)
	end
	loops_expanded = macroexpand(MRFingerprinting, loops)
	return quote
		@assert all(shape .> 0)
		for (arr, off) in ((dest, offset_dest), (src, offset_src))
			@assert all(0 .< off .+ shape .≤ size(arr))
			@assert all(0 .≤ off .≤ size(arr))
		end
		@nextract $N k_dest offset_dest
		@nextract $N k_src offset_src
		@nexprs $N d -> i_max_d = shape[d]
		$(Expr(:macrocall, Symbol("@tturbo"), "", loops_expanded.args[2].args[2]))
		return dest
	end
end

