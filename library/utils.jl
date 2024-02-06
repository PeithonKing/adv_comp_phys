function differentiate(f, x, h=1e-6)
    return (f(x + h/2) - f(x - h/2)) / h
end

function truncate_to_decimal_places(number, decimal_places)
    scale = 10 ^ decimal_places
    truncated_number = floor(number * scale) / scale
    return string(truncated_number)
end