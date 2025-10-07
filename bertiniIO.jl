function read_real_parts(filename::String)
    # Read all nonempty lines
    lines = filter(!isempty, strip.(readlines(filename)))

    # First line: number of points (may not be needed)
    n_points = parse(Int, split(lines[1])[1])

    # The rest: numeric data
    data_lines = lines[2:end]

    # Parse each line, taking only the first number (real part)
    real_parts = [parse(Float64, split(line)[1]) for line in data_lines]

    # Each block corresponds to 4 complex numbers → reshape
    m = 4  # number per block
    if length(real_parts) % m != 0
        @warn "Number of lines not divisible by 4 — check the file structure"
    end

    # Return as a matrix: each row = 4-tuple
    return reshape(real_parts, m, :)'
end