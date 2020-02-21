using .Measurements: Measurements

Base.convert(::Type{Measurements.Measurement}, y::NeuralQuantum.Measurement) =
    Measurements.measurement(value(y), uncertainty(y))

Measurements.Measurement(y::NeuralQuantum.Measurement) =
    Measurements.measurement(y)

Measurements.measurement(y::NeuralQuantum.Measurement) =
    Measurements.measurement(value(y), uncertainty(y))
