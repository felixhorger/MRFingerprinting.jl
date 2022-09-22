
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
