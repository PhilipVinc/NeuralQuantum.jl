abstract type Comm end

struct AllThreadComm{A,B,C,D} <: Comm
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
    return AllThreadComm(
        Threads.Atomic{Int}(-Threads.nthreads()),
        Barrier(Threads.nthreads()),
        zeros(UInt, Threads.nthreads()),
        Threads.nthreads())
end

Comm_size(t::AllThreadComm) = Threads.nthreads()
Comm_rank(t::AllThreadComm) = Threads.threadid()

struct SetThreadComm{A,B,C,D} <: Comm
    status::A
    barrier::D
    passthrough::B
    nthreads::C
    thread_id::C
end

function Comms(n::Int)
    st  = Threads.Atomic{Int}(n)
    bar = Barrier(n)
    pas = zeros(UInt, n)
    return tuple((SetThreadComm(
        st, bar, pas,
        n, i) for i=1:n)...)
end

Comm_size(t::SetThreadComm) = t.nthreads
Comm_rank(t::SetThreadComm) = t.thread_id
