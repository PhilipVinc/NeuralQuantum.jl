struct Comm{A,B,C,D}
    status::A
    barrier::D
    passthrough::B
    nthreads::C
end

"""
    TPI.Comm()

Builds a communicator for Thread-inter communication.
"""
function Comm()
    return Comm(
        Threads.Atomic{Int}(-Threads.nthreads()),
        Barrier(Threads.nthreads()),
        zeros(UInt, Threads.nthreads()),
        Threads.nthreads())
end
