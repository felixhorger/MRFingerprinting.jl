
using Revise
using BenchmarkTools
using FFTW
using Cthulhu
import MRIRecon
import MRFingerprinting as MRF
import MRITrajectories
import PyPlot as plt
import PyPlotTools

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
ym = Vector{ComplexF64}(undef, num_columns * num_angles * num_channels * num_σ);
y = rand(ComplexF64, num_columns, num_angles, num_channels, num_σ);
y_vec = vec(y)
lr_mix_d = MRF.decomplexify(lr_mix)
lr_mix_d_c = MRF.decomplexify(convert.(ComplexF64, lr_mix))
#
MRF.apply_lowrank_mixing!(ym, y_vec, lr_mix_d, num_columns, num_channels)
MRF.apply_lowrank_mixing!(ym, y_vec, lr_mix_d, num_columns, num_channels)
#
L = MRF.plan_lr2time(transpose(VH), conj.(VH), num_columns * num_angles * num_channels)
M = MRIRecon.plan_masking(spoke_indices, (num_columns, num_angles, num_channels, num_time))
#
# Non-optimised method
ym2 = L' * M * L * y_vec
@assert ym2 ≈ ym



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
num_channels = 10 * 192
#
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

T = MRF.plan_lowrank_toeplitz_embedding(y, F_double_fov, lr_mix)
GC.gc(true)

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

@btime ($T * $vec_x) samples=3;
@btime ($A * $vec_x) seconds=3;


x_padded, vec_x_padded, F, FH_unnormalised, M, centre_indices = MRF.prepare_lowrank_toeplitz_embedding(F_double_fov, lr_mix, shape, num_channels)

MRF.apply_lowrank_toeplitz_embedding(
	vec_x,
	x_padded,
	vec_x_padded,
	vec_y,
	F,
	FH_unnormalised,
	M,
	centre_indices
) 

