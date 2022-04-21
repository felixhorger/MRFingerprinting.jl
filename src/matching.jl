
"""
	closest(params::AbstractVector{<: Number}, target::AbstractVector{<: Number})

Find indices of values in `target` which are closest to arbitrarily valued parameters `params`.
Note that `target` must be sorted in ascending order.

"""
function closest(params::AbstractVector{<: Number}, target::AbstractVector{<: Number})
	c = Vector{Int64}(undef, length(params))
	p = sortperm(params)
	i = 1
	j = 1
	i_max = length(params)
	j_max = length(target)
	@inbounds while j <= j_max && i <= i_max
		k = p[i]
		if params[k] < target[j]
			c[k] = j
			i += 1
		else
			j += 1
		end
	end
	if i <= i_max
		c[i:end] .= length(target)
	end
	return c
end


"""
	match2maps(matches::AbstractVector{<: Integer}, parameters::NTuple{N, AbstractVector}) where N

Assign the respective parameter value from `parameters` corresponding to the matching index specified in `matches`.
Returns a `Vector` of `Vector`s, one for each parameter.

"""
function match2maps(matches::AbstractVector{<: Integer}, parameters::NTuple{N, AbstractVector}) where N
	# Start with inner most parameter
	parameters_lengths = length.(parameters)
	maps = Vector{Vector}(undef, length(parameters))
	matches = matches .- 1 # This copies
	indices = similar(matches)
	for p in eachindex(parameters)
		@. indices = mod(matches, parameters_lengths[p]) + 1
		maps[p] = parameters[p][indices]
		@. matches = matches ÷ parameters_lengths[p]
	end
	return maps
end

"""
	unpack_parameters(map::AbstractVector{<: NTuple{N, <: Any}}) where N

Unpack `Tuple`-like parameter maps, i.e. with parameter combinations, into maps of individual elements.

"""
function unpack_parameters(map::AbstractVector{<: NTuple{N, <: Any}}) where N
	ntuple( i -> [map[j][i] for j in eachindex(map)], N )
end


"""
	prepare_matching(D::AbstractMatrix{<: Real}, f::AbstractMatrix{<: C}) where C <: Complex

Returns
- `matches`
- `match_overlap`
- `fd`

"""
function prepare_matching(
	D::AbstractMatrix{<: Real},
	f::AbstractMatrix{<: C}
) where C <: Complex
	# Check arguments
	num_D = size(D, 1)
	timepoints = size(D, 2)
	@assert size(f, 1) == timepoints
	num_f = size(f, 2)

	# Allocate memory
	matches = Vector{Int64}(undef, num_f)
	match_overlap = zeros(Float64, num_f)

	# Turn f into an array of two Reals per Complex
	fd = decomplexify(f)
	return matches, match_overlap, fd
end
function prepare_matching(
	D::AbstractMatrix{<: Real},
	v::NTuple{2, AbstractMatrix{<: Complex}}
)
	f, g = v
	matches, match_overlap, fd = prepare_matching(D, f)
	@assert size(g) == size(f)
	# Alike f, turn g also into an array of two Reals per Complex
	gd = decomplexify(g)

	return matches, match_overlap, (fd, gd)
end
function prepare_matching(f::AbstractMatrix{C}) where C <: Complex
	Array{real(C), 3}(undef, 2, size(f, 1), size(f, 2))
end
# Note: the type and dimension checks are performed in the above prepare_matching used for full dictionary matching
function prepare_matching(
	D::AbstractMatrix{<: Real},
	f::AbstractMatrix{C},
	indices::AbstractVector{<: Integer},
	stride::Integer
) where C <: Complex
	@assert all(1 .<= indices .<= size(D, 1) ÷ stride)
	@assert length(indices) == size(f, 2)
	subset = Vector{Int64}(undef, size(f, 2))
	f_subset = prepare_matching(f)
	return f_subset, subset
end
function prepare_matching(
	D::AbstractMatrix{<: Real},
	v::NTuple{2, AbstractMatrix{C}},
	indices::AbstractVector{<: Integer},
	stride::Integer
) where C <: Complex
	f, g = v
	f_subset, subset = prepare_matching(D, f, indices, stride)
	g_subset = prepare_matching(f)
	return (f_subset, g_subset), subset
end



function find_maximum_overlap!(
	matches::AbstractVector{<: Integer}, # Indices of matches in D
	match_overlap::AbstractVector{<: Real}, # Values of the overlap
	overlap::AbstractMatrix{<: Real},
	f_indices::AbstractVector{<: Integer},
	d0::Integer=0
)
	@inbounds for (i, fi) in enumerate(f_indices)
		# iterate fingerprints in d
		overlap_with_match = 0.0
		match_index = 0 # if overlap is zero, then no fingerprint is matching. this means the fingerprint in f is zero.
		for di in axes(overlap, 1)
			# if greater overlap, store index and value
			this_overlap = overlap[di, i]
			if overlap_with_match < this_overlap
				match_index = di
				overlap_with_match = this_overlap
			end
		end
		match_overlap[fi] = overlap_with_match
		matches[fi] = d0 + match_index
	end
	return
end



# Comments I need to filter:
# Full
# TODO: Is it better to have whole dictionary and few fingerprints or a medium amount of both?
# In the first case, the algorithm is simpler, but in the second more can be done while keeping everything on the stack.
# Dictionary must be normalised
# Dictionary must be real, so that offset phase is in f (this is always possible!)
# fingerprints f[time, fingerprint]
# dictionary D[fingerprint, time]

# Equally in the SVD domain
# fingerprints f[singular component, fingerprint] = V^H * fingerprints
# dictionary D[fingerprint, singular component] = U * S / ||U * S||_2

# Subdictionary
# Match with a part of the dictionary based on known parameter values
# Dictionary must be normalised
# fingerprints f[time, fingerprint]
# dictionary D[fingerprint, time]

# f[singular component, fingerprint] = V^H * f
# D[fingerprint, singular component] = U * S / ||U * S||_2

# indices corresponding to known parameters of the fingerprints in the dictionary.
# The known parameter(s) have to be the slowest changing ones in the first axis of D.
# stride is the number of fingerprints in subdictionaries
# step how many fingerprints to process at the same time (maximum, might be less)




#=
	Define matching helper functions for
	1) ||d ⋅ f||^2 and
	2) ||d ⋅ f||^2 + 2Re{||g ⋅ d * d ⋅ f||}
=#

# Helpers for matching with a whole dictionary
@generated function match!(
	matches::AbstractVector{<: Integer},	# Modified
	match_overlap::AbstractVector{<: Real}, # |
	overlap::AbstractMatrix{<: Real}, 		# |
	D::AbstractMatrix{<: Real},
	v::Union{AbstractArray{<: Real, 3}, NTuple{2, AbstractArray{<: Real, 3}}},
	step::Integer
)
	if v <: AbstractArray{<: Real, 3}
		# only f
		expand_v = :(f = v)
		args = :(f[:, :, i:fi_max+i-1])
	else
		# f and g
		expand_v = :(f, g = v)
		args = :(f[:, :, f_indices], g[:, :, f_indices])
	end

	return quote
		$expand_v
		num_f = size(f, 3)
		fi_max = step
		@inbounds @views for i = 1:step:num_f
			if (i + step) > (num_f + 1) # Moved the one from left to right
				fi_max = num_f - i + 1
			end
			f_indices = i:fi_max+i-1
			overlap!(overlap, D, $args)
			find_maximum_overlap!(matches, match_overlap, overlap, f_indices)
		end
		return
	end
end


# Helpers for matching where fingerprints are randomly extracted from f (or g)
@generated function match!(
	matches::AbstractVector{<: Integer},	# Modified
	match_overlap::AbstractVector{<: Real},
	overlap::AbstractMatrix{<: Real},
	v_subset::Union{AbstractArray{<: Real, 3}, NTuple{2, AbstractArray{<: Real, 3}}}, # up to here incl.
	D::AbstractMatrix{<: Real},
	v::Union{AbstractArray{<: Real, 3}, NTuple{2, AbstractArray{<: Real, 3}}},
	f_indices::AbstractVector{<: Integer},
	step::Integer,
	d0::Integer
)

	if v <: AbstractArray{<: Real, 3}
		# only f
		@assert v_subset <: AbstractArray{<: Real, 3}
		expand_v = quote
			f = v
			f_subset = v_subset
		end
		extract_f = :(f_subset[:, :, this_step] .= f[:, :, this_f_indices])
		args = :(f_subset[:, :, this_step])
	else
		# f and g
		@assert v_subset <: NTuple{2, AbstractArray{<: Real, 3}}
		expand_v = quote
			f, g = v
			f_subset, g_subset = v_subset
		end
		extract_f = quote
			f_subset[:, :, this_step] .= f[:, :, this_f_indices]
			g_subset[:, :, this_step] .= g[:, :, this_f_indices]
		end
		args = :(f_subset[:, :, this_step], g_subset[:, :, this_step])
	end

	return quote
		$expand_v
		num_f = length(f_indices)
		fi_max = step
		@inbounds @views for i = 1:step:num_f
			if (i + step) > (num_f + 1) # Moved the one from left to right
				fi_max = num_f - i + 1
			end
			this_step = i:fi_max+i-1
			this_f_indices = f_indices[this_step]
			$extract_f
			overlap!(overlap, D, $args)
			find_maximum_overlap!(matches, match_overlap, overlap, this_f_indices, d0)
		end
		return
	end
end

# The for loop over sub-dictionaries
function match!(
	matches::AbstractVector{<: Integer}, # Modified
	match_overlap::AbstractVector{<: Real},
	overlap::AbstractMatrix{<: Real},
	v_subset::Union{AbstractArray{<: Real, 3}, NTuple{2, AbstractArray{<: Real, 3}}},
	subset::AbstractVector{<: Integer}, # up to here incl.
	D::AbstractMatrix{<: Real},
	v::Union{AbstractArray{<: Real, 3}, NTuple{2, AbstractArray{<: Real, 3}}},
	indices::AbstractVector{<: Integer}, # Known parameter indices
	stride::Integer,
	step::Integer
)
	@inbounds for s in 1:stride:size(D, 1)
		# Find the subset of fingerprints which have this current parameter index assigned
		num_subset_f = 0
		parameter_index = (s-1) ÷ stride + 1
		for i in eachindex(indices)
			if indices[i] == parameter_index
				num_subset_f += 1
				subset[num_subset_f] = i
			end
		end
		num_subset_f == 0 && continue

		@views match!(
			matches, match_overlap, overlap, v_subset,
			D[s:s+stride-1, :], v,
			subset[1:num_subset_f],
			step,
			s-1
		)
	end
	return
end


# Full dictionary matching
function match(
	D::AbstractMatrix{<: Real},
	v::Union{AbstractArray{<: Complex}, NTuple{2, AbstractArray{<: Complex}}},
	step::Integer
)
	# Returns matching indices and overlap
	matches, match_overlap, vd = prepare_matching(D, v)
	overlap = Matrix{Float64}(undef, size(D, 1), step)
	match!(matches, match_overlap, overlap, D, vd, step)
	return matches, match_overlap
end


# Sub-dictionary matching
function match(
	D::AbstractMatrix{<: Real},
	v::Union{AbstractMatrix{<: Complex}, NTuple{2, AbstractArray{<: Complex}}},
	indices::AbstractVector{<: Integer},
	stride::Integer,
	step::Integer
)
	# Check arguments, allocate memory, prepare arrays
	matches, match_overlap, vd = prepare_matching(D, v)
	overlap = Matrix{Float64}(undef, stride, step)
	v_subset, subset = prepare_matching(D, v, indices, stride)
	match!(
		matches, match_overlap, overlap, v_subset, subset,
		D, vd,
		indices, stride, step
	)
	return matches, match_overlap
end

