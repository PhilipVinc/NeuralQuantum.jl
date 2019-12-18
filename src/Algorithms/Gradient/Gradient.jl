export Gradient

################################################################################
######   Preconditioning algorithm definition (structure holding params)  ######
################################################################################
"""
    Gradient()

Algorithm for descending along the steepest gradient with SGD-based optimizers.
"""
struct Gradient <: Algorithm end
