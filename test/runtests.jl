using Revise
using BenchmarkTools
using FFTW
using Cthulhu
using PlasticArrays
import MRIRecon
import MRFingerprinting as MRF
import MRITrajectories
import PyPlot as plt
import PyPlotTools



# Projection
MRF.plan_dictionary_projection(
	DT_renorm::AbstractMatrix{<: Real},
	matches::AbstractVector{<: Integer},
	num_x::Integer,
	num_σ::Integer;
	dtype::Type{C}=ComplexF64,
	out::AbstractVector{<: C}=empty(Vector{dtype})
)




# Testing closest()
MRF.closest([2.0, 3.4], [2, 3, 4, 5]) # (2, 3), (3, 2)
MRF.closest([(2.0, 2.0), (2.0, 3.5)], [(2.0, 2.0), (2., 4.), (3., 2.)]) # (2, 3), (3, 2)


# Haar wavelets
num_σ = 5
num_time = 313
VH = zeros(num_σ, num_time)
n1 = 1 / sqrt(num_time)
n2 = 1 / sqrt(num_time-1)
n3 = sqrt(2) * n2
n4 = 0.5
VH[1, :] .= n1
VH[2, 1:156] .= n2
VH[2, 157:end-1] .= -n2
VH[3, 1:78] .= n3
VH[3, 79:156] .= -n3
VH[4, 157:235] .= n3
VH[4, 235:end-1] .= -n3
VH[5, 153:154] .= n4
VH[5, 155:156] .= -n4
VH * VH'

let
	# Low-rank mixing compared to conventional L' * M * L'
	# Choose small matrix size so that the time vector fits into memory
	num_columns = 16
	num_lines = 16
	shape = (num_columns, num_lines)
	num_angles = MRITrajectories.required_num_spokes(num_lines)
	#
	spokes_per_timepoint = 2
	total_num_spokes = spokes_per_timepoint * num_time
	φ, spoke_indices = MRITrajectories.golden_angle_incremented(total_num_spokes, num_angles)
	φ, spoke_indices = MRITrajectories.sort_angles!(φ, spoke_indices, spokes_per_timepoint, num_time)
	spoke_indices = CartesianIndex.(MRITrajectories.chronological_order(spoke_indices))
	lr_mix = MRF.lowrank_mixing(VH, spoke_indices, (num_angles,))
	num_channels = 10
	#
	# Low-rank mixing
	ym = Array{ComplexF64, 4}(undef, num_columns, num_angles, num_channels, num_σ);
	y = rand(ComplexF64, num_columns, num_angles, num_channels, num_σ);
	y_vec = vec(y)
	lr_mix_d = MRF.decomplexify(lr_mix)
	lr_mix_d_c = MRF.decomplexify(convert.(ComplexF64, lr_mix))
	#
	MRF.apply_lowrank_mixing!(ym, y, lr_mix_d, num_columns, num_channels)
	MRF.apply_lowrank_mixing!(ym, y, lr_mix_d_c, num_columns, num_channels)
	# Non-optimised method
	L = MRF.plan_lr2time(transpose(VH), conj.(VH), num_columns * num_angles * num_channels)
	M = MRIRecon.plan_masking!(spoke_indices, (num_columns, num_angles, num_channels, num_time))
	#
	ym2 = L' * M * L * y_vec
	# Check equality
	@assert ym2 ≈ vec(ym)
end
GC.gc(true)


# Low-rank sparse to dense
num_time = 41
VH = rand(6, num_time)
V_conj = conj.(VH')
shape = (128, 128)
#128 * 4 * prod(shape) * num_time * 16 * 1e-9
N = 70000
sampling = MRITrajectories.uniform_dynamic(shape, num_time, 10000)[1:N]
x = rand(128, 4, N)
dense_kt = MRIRecon.sparse2dense(x, sampling, shape, num_time);
dense_kσ = MRF.time2lr(dense_kt, V_conj);
direct_dense_kσ = MRF.lowrank_sparse2dense(x, sampling, shape, V_conj);
parallel_dense_kσ = MRF.lowrank_sparse2dense_parallel(x, sampling, shape, V_conj);
@assert isapprox(direct_dense_kσ, dense_kσ; atol=1e-12)
@assert isapprox(direct_dense_kσ, parallel_dense_kσ; atol=1e-16)
#imshow(abs.(dense_kσ[1, 1, :, :, :]))
#imshow(abs.(direct_dense_kσ[1, 1, :, :, :]))

# Performance
num_time = 313
VH = rand(6, num_time)
V_conj = conj.(VH')
shape = (256, 256)
sampling = MRITrajectories.uniform((shape..., num_time))
N = 70000
sampling = [CartesianIndex(s[1], s[2]) for s in sampling[1:N]]
x = zeros(ComplexF64, 256, 20, N) # readout, channel, k(sparse)-t
dense_kt = MRF.time2lr(dense_kt, V_conj);
dense_kt = MRIRecon.sparse2dense(x, sampling, shape, num_time);

@time direct_dense_kσ = MRF.lowrank_sparse2dense(x, sampling, shape, V_conj);

@time parallel_dense_kσ = MRF.lowrank_sparse2dense_parallel(x, sampling, shape, V_conj);

#@code_warntype MRF.lowrank_sparse2dense_parallel(x, sampling, shape, V_conj);

# sparse kt to lowrank operator test
num_time = 13
VH = rand(6, num_time)
V_conj = conj.(VH')
shape = (128, 128)
sampling = repeat(collect(vec(CartesianIndices(shape))), inner=num_time)
x = rand(ComplexF64, 1, shape..., 1, 6)
xt = MRF.lr2time(x, VH)
y = MRF.time2lr(xt, V_conj)

UL = MRF.plan_lowrank2sparse(V_conj, VH, sampling, 1, shape, 1)

z = permutedims(reshape(UL * vec(x), 1, num_time, shape..., 1), (1, 3, 4, 5, 2))
@assert maximum(abs, z .- xt) < 1e-10

y_ = reshape(UL' * (UL * vec(x)), size(y))
@assert maximum(abs, y_ .- y) < 1e-10




# Low-rank Toeplitz Embedding
num_columns = 320
num_lines = 320
shape = (num_columns, num_lines)
num_angles = 400 #MRITrajectories.required_num_spokes(num_lines) # Use less spokes than requested by num_lines
k = MRITrajectories.radial_spokes(num_angles, num_columns)
k = reshape(k, 2, num_columns * num_angles)

spokes_per_timepoint = 2
total_num_spokes = spokes_per_timepoint * num_time
φ, spoke_indices = MRITrajectories.golden_angle_incremented(total_num_spokes, num_angles)
φ, spoke_indices = MRITrajectories.sort_angles!(φ, spoke_indices, spokes_per_timepoint, num_time)
spoke_indices = CartesianIndex.(MRITrajectories.chronological_order(spoke_indices))
lr_mix = MRF.lowrank_mixing(VH, spoke_indices, (num_angles,))
num_channels = 1 * 192

F_double_fov = MRIRecon.plan_fourier_transform(k[1, :], k[2, :], (2num_columns, 2num_lines, 1); modeord=1)
F = MRIRecon.plan_fourier_transform(k[1, :], k[2, :], (num_columns, num_lines, num_channels * num_σ))
M = MRF.plan_lowrank_mixing(lr_mix, num_columns, num_channels)
A = F' * M * F

x = zeros(ComplexF64, num_columns, num_lines, num_channels, num_σ);
x[num_columns÷2, num_lines÷2, :, :] .= 1;
vec_x = vec(x)
y = zeros(ComplexF64, num_columns, num_lines, num_channels, num_σ);
vec_y = vec(y)
GC.gc(true)

T = MRF.plan_lowrank_toeplitz(y, F_double_fov, lr_mix)
GC.gc(true)

@benchmark reshape($T * $vec_x, $num_columns, $num_lines, $num_channels, $num_σ) samples=1 seconds=40 evals=1
@benchmark reshape($A * $vec_x, $num_columns, $num_lines, $num_channels, $num_σ) samples=4 seconds=40 evals=1

@time z1 = reshape(T * vec_x, num_columns, num_lines, num_channels, num_σ);
@time z2 = reshape(A * vec_x, num_columns, num_lines, num_channels, num_σ);

@assert isapprox(z1, z2; rtol=1e-7)
GC.gc(true)


fig, axs = plt.subplots(3, 5, sharex=true, sharey=true)
@views for σ = 1:num_σ
	@views begin
		a1 = 1e3 .* abs.(z1[:, :, 1, σ])
		a2 = 1e3 .* abs.(z2[:, :, 1, σ])
		a3 = 1e3 .* abs.(z1[:, :, 1, σ] .- z2[:, :, 1, σ])
	end
	vmax = max(maximum(a1), maximum(a2))
	vmin = min(minimum(a1), minimum(a2))
	image = axs[1, σ].imshow(a1; vmin, vmax)
	PyPlotTools.add_colourbar(fig, axs[1, σ], image)
	image = axs[2, σ].imshow(a2; vmin, vmax)
	PyPlotTools.add_colourbar(fig, axs[2, σ], image)
	image = axs[3, σ].imshow(a3; vmin=0, vmax=vmax*1e-8)
	PyPlotTools.add_colourbar(fig, axs[3, σ], image)
end


# Plastic arrays
sp = MRF.lowrank_toeplitz_padded_size((num_columns, num_lines), num_channels, num_σ)
ap_in = PlasticArray(prod(sp) * sizeof(ComplexF64))
ap_out = PlasticArray(prod(sp) * sizeof(ComplexF64))
GC.gc(true)

Tp = MRF.plan_lowrank_toeplitz!(
	(num_columns, num_lines), num_channels,
	F_double_fov,
	lr_mix;
	x_padded=mould(ap_out, ComplexF64, sp), # input to operator is ap_in
	y_padded=mould(ap_in, ComplexF64, sp),
)
z3 = reshape(Tp * copy(vec_x), num_columns, num_lines, num_channels, num_σ);
@assert z3 ≈ z1

Fsp = MRIRecon.fourier_transform_size(k[1, :], k[2, :], (num_columns, num_lines, num_channels * num_σ))
# (out) Fp' (in) * Mp (out) * Fp
Fp = MRIRecon.plan_fourier_transform(
	k[1, :], k[2, :],
	(num_columns, num_lines, num_channels * num_σ);
	Fx=mould(ap_out, ComplexF64, sp[1]),
	FHy=mould(ap_in, ComplexF64, sp[2])
)
GC.gc(true)
Msp = MRF.lowrank_mixing_dim(num_columns, num_angles, num_channels, num_σ)
Mp = MRF.plan_lowrank_mixing(
	lr_mix,
	num_columns, num_channels;
	Mx=mould(ap_in, ComplexF64, Msp)
)
GC.gc(true)

Ap = Fp' * Mp * Fp

@time z4 = reshape(Ap * vec_x, num_columns, num_lines, num_channels, num_σ);
@assert z4 ≈ z2


#= OLD

let v = vec(ones(ComplexF64, (1, num_lines, num_partitions, num_channels, num_σ)))
	a = reshape(M * v, num_lines, num_partitions, num_channels, num_σ)
	b = reshape(Λ' * DHD * Λ * v, num_lines, num_partitions, num_channels, num_σ)
	#plt.figure()
	#plt.imshow(abs.(a[:, :, 1, 1]))
	#plt.figure()
	#plt.imshow(abs.(b[:, :, 1, 1]))
	#plt.show()
	#println(a[1:10], b[1:10])
	@assert a ≈ b
end

=#

