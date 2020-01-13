abstract type Comm

struct AllThreadComm{A,B,C,D}
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

struct SetThreadComm{A,B,C,D}
    status::A
    barrier::D
    passthrough::B
    nthreads::C
    thread_id::C
end

function Comm(i::Int, n::Int)
    return AllThreadComm(
        Threads.Atomic{Int}(-Threads.nthreads()),
        Barrier(Threads.nthreads()),
        zeros(UInt, Threads.nthreads()),
        n,
        i)
end

Comm_size(t::SetThreadComm) = t.nthreads
Comm_rank(t::SetThreadComm) = t.thread_id
