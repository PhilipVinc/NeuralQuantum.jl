import .TensorBoardLogger: TensorBoardLogger

function TensorBoardLogger.preprocess(name, val::Measurement, data)
    TensorBoardLogger.preprocess(name*"/mean", val.mean, data)
    TensorBoardLogger.preprocess(name*"/err", val.error, data)

    return data
end
