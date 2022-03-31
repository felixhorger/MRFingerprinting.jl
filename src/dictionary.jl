
function normalise(D::AbstractMatrix{<: Number})::Tuple{AbstractMatrix{<: Number}, Vector{Float64}}
	# Axes: D[fingerprint, time]
	norms = sqrt.(dropdims(sum(conj.(D) .* D; dims=2); dims=2))
	D ./= norms
	return D, norms
end

function compress(D::AbstractMatrix{<: Number})::Tuple{Matrix{<: Number}, Vector{<: Number}, Matrix{<: Number}}
	# Axes: D[fingerprint, time]
	# Returns:
	# compressed_dictionary[fingerprint, singular_component]
	# singular values σ
	# transformation matrix V^h[singular_component, time]
	F = svd(D)
	return F.U * diagm(F.S), F.S, F.Vt
end

function compression_error(D::AbstractMatrix{<: Number}, Vh::AbstractMatrix{<: Number})::Vector{Float64}
	# Dictionary must be normalised
	# Axes:
	# D[fingerprint, time]
	# Vh[singular component, time]
	# Singular components must be selected before passing Vh to this function

	# Project there and back again (an expected journey)
	D_approx = *(D, Vh', Vh) # This picks the most efficient order of multiplication
	# Compute sum of squares error
	error = D_approx .- D
	return sqrt.( sum(conj.(error) .* error; dims=2) )
end

function overlap(D::AbstractMatrix{<: Number})::Real
	# Dictionary must be normalised
	# D[fingerprint, time]
	# It's better to have the small axis on the inside of the multiplication
	abs.(D' * D)
end

function energy_fraction(σ::AbstractVector{<: Number})
	σ2 = abs2.(σ);
	frac = cumsum(σ2) ./ sum(σ2);
end


function match(D::AbstractMatrix{<: Real}, f::AbstractMatrix{<: Complex}, step::Integer)
	# TODO: Is it better to have whole dictionary and few fingerprints or a medium amount of both?
	# In the first case, the algorithm is simpler, but in the second more can be done while keeping everything on the stack.
	# Dictionary must be normalised
	# Dictionary must be real, so that offset phase is in f (this is always possible!)
	# fingerprints f[time, fingerprint]
	# dictionary D[fingerprint, time]

	# Equally in the SVD domain
	# fingerprints f[singular component, fingerprint] = V^H * fingerprints
	# dictionary D[fingerprint, singular component] = U * S / ||U * S||_2

	# Returns matching indices and overlap

	num_D = size(D, 1)
	timepoints = size(D, 2)
	num_f = size(f, 2)
	# Allocate memory
	matches = Vector{Int64}(undef, num_f)
	match_overlap = zeros(Float64, num_f)
	overlap = Matrix{Float64}(undef, num_D, step)

	# Iterate blocks of fingerprints with size step
	fi_max = step
	f_destruct = reinterpret(real(eltype(f)), f)
	for i = 1:step:num_f
		if (i + step) > (num_f + 1) # Moved the 1 from left to right
			fi_max = num_f - i + 1
		end
		@turbo for di = 1:num_D, fi = 1:fi_max
			v_real = 0.0
			v_imag = 0.0
			for t = 1:timepoints
				v_real += D[di, t] * f_destruct[2*t - 1,	fi + i - 1]
				v_imag += D[di, t] * f_destruct[2*t,		fi + i - 1]
			end
			overlap[di, fi] = v_real^2 + v_imag^2
		end
		# Iterate fingerprints in f
		@inbounds for fi in 1:fi_max
			fi_global = i + fi - 1 # Actual index in the array f
			# Iterate fingerprints in D
			overlap_with_match = 0.0
			match_index = 0
			for di = 1:num_D
				# If greater overlap, store index and value
				this_overlap = overlap[di, fi]
				if overlap_with_match < this_overlap
					match_index = di
					overlap_with_match = this_overlap
				end
			end
			# If overlap is zero, then no fingerprint is matching
			if overlap_with_match == 0
				match_index = 0
			end
			match_overlap[fi_global] = overlap_with_match
			matches[fi_global] = match_index
		end
	end
	match_overlap .= sqrt.(match_overlap ./ dropdims(sum(abs2.(f); dims=1); dims=1)) # Because the square of that is used before
	return matches, match_overlap
end

function closest(params::AbstractVector{<: Number}, target::AbstractVector{<: Number})
	# Find indices of parameters of the dictionary which are closest to arbitrarily valued parameter (e.g measured)
	# params: known parameters for each fingerprint
	# target: parameters of the dictionary, must be sorted!
	c = Vector{Int64}(undef, length(params))
	p = sortperm(params)
	i = 1
	j = 1
	i_max = length(params)
	j_max = length(target)
	while j <= j_max && i <= i_max
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

function match(
	D::AbstractMatrix{<: Real},
	f::AbstractMatrix{<: Complex},
	indices::AbstractVector{<: Integer},
	stride::Integer,
	step::Integer
)
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

	num_D = size(D, 1)
	timepoints = size(D, 2)
	num_f = size(f, 2)

	# Check indices
	@assert all(1 .<= indices .<= num_D ÷ stride)

	# Allocate memory
	matches = Vector{Int64}(undef, num_f)
	match_overlap = zeros(Float64, num_f)
	overlap = Matrix{Float64}(undef, num_D, step)
	subset = Vector{Int64}(undef, num_f)

	f_destruct = reinterpret(real(eltype(f)), f)
	for s in 1:stride:num_D
		# Find the subset of fingerprints which have this current parameter index assigned
		num_subset_f = 0
		parameter_index = (s-1) ÷ stride + 1
		@inbounds for i in eachindex(indices)
			if indices[i] == parameter_index
				num_subset_f += 1
				subset[num_subset_f] = i
			end
		end
		num_subset_f == 0 && continue
		# Do the matching for this subset, iterate through vector "subset" in blocks of size "step"
		fi_max = step
		for i = 1:step:num_subset_f
			if (i + step) > (num_subset_f + 1) # Moved the one from left to right
				fi_max = num_subset_f - i + 1
			end
			@turbo for di = s:s+stride-1, fi = 1:fi_max
				v_real = 0.0
				v_imag = 0.0
				for t = 1:timepoints
					v_real += D[di, t] * f_destruct[2*t - 1,	subset[i+fi-1]]
					v_imag += D[di, t] * f_destruct[2*t,		subset[i+fi-1]]
				end
				overlap[di, fi] = v_real^2 + v_imag^2
			end
			# Iterate fingerprints in f
			@inbounds for fi in 1:fi_max
				fi_global = subset[i+fi-1] # Actual index in the array f
				# Iterate fingerprints in D
				overlap_with_match = 0.0
				match_index = 0 # If overlap is zero, then no fingerprint is matching
				for di = 1:num_D
					# If greater overlap, store index and value
					this_overlap = overlap[di, fi]
					if overlap_with_match < this_overlap
						match_index = di
						overlap_with_match = this_overlap
					end
				end
				match_overlap[fi_global] = overlap_with_match
				matches[fi_global] = match_index
			end
		end
	end
	match_overlap .= sqrt.(match_overlap ./ dropdims(sum(abs2.(f); dims=1); dims=1)) # Because the square of that is used before
	return matches, match_overlap
end


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

function unpack_parameters(map::AbstractVector{<: NTuple{N, <: Any}}) where N
	ntuple( i -> [map[j][i] for j in eachindex(map)], N )
end

