
function normalise(D::AbstractMatrix{<: Number})::Tuple{AbstractMatrix{<: Number}, Vector{Float64}}
	# Axes: D[fingerprint, time]
	norms = sqrt.(sum(conj.(D) .* D; dims=2))
	D ./= norms
	return D, norms
end

function compress(D::AbstractMatrix{<: Number})::Tuple{Matrix{<: Number}, Vector{<: Number}, Matrix{<: Number}}
	# Axes: D[fingerprint, time]
	# Returns: compressed dictionary, singular values, transformation matrix V^h
	F = svd(D)
	return F.S, diag(F.S), F.Vt
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
	# Apparently it's faster to have the large axis on the inside of the multiplication
	abs(dictionary' * dictionary)
end

function energy_fraction(σ::AbstractVector{<: Number})
	sigmas_sq = conj(sigmas) .* sigmas;
	frac = cumsum(sigmas_sq) ./ sum(sigmas_sq);
end
