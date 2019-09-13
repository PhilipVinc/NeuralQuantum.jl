function TensorBoardLogger.preprocess(name, val::SampledObservables, data)
    for (key,val)=zip(val.ObsNames, val.ObsAve)
        TensorBoardLogger.preprocess(name*"/"*string(key), real(val), data)
    end
    data
end

#=function ValueHistoriesLogger.preprocess(name, val::SampledObservables, data)
    for (key,val)=zip(val.ObsNames, val.ObsAve)
        ValueHistoriesLogger.preprocess(name*"/"*string(key), real(val), data)
    end
    data
end=#

function TensorBoardLogger.preprocess(name, val::SREvaluation, data)
    TensorBoardLogger.preprocess(name*"/Loss", val.L, data)
    data
end

#=function ValueHistoriesLogger.preprocess(name, val::SREvaluation, data)
    ValueHistoriesLogger.preprocess(name*"/Loss", val.L, data)
    ValueHistoriesLogger.preprocess(name*"/Loss_Convergence", val.LVals, data)
    data
end=#

function TensorBoardLogger.preprocess(name, val::NeuralNetwork, data)
    data
end

#=function ValueHistoriesLogger.preprocess(name, val::NeuralNetwork, data)
    push!(data, name=>val)
    data
end=#
