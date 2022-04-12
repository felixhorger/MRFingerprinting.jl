
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

@inline function normalise_overlap!(overlap::AbstractVector{<: Real}, f::AbstractMatrix{<: Number})
	f_norm_squared = vec(sum(abs2, f; dims=1))
	overlap .= sqrt.(overlap ./ f_norm_squared)
end

function energy_fraction(σ::AbstractVector{<: Number})
	σ2 = abs2.(σ);
	frac = cumsum(σ2) ./ sum(σ2);
end

