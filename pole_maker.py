N = 30
spacing = 0.1  # meters between poles
x_start = -(N - 1) / 2 * spacing

for i in range(N):
    x = x_start + i * spacing
    print(f'<body><geom type="cylinder" pos="{x:.2f} 0 0.5" size="0.05 0.5" rgba="1 0.3 0.3 1"/></body>')
