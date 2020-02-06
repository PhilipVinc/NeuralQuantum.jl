"""
    Barrier(num_threads)
Create a thread-safe barrier that blocks all tasks until `num_threads`
tasks call `sync`.
"""
mutable struct Barrier
    notify::Threads.Condition
    tar_cnt::Int
    cur_cnt::Int
    Barrier(n_threads) = n_threads > 0 ? new(Threads.Condition(), n_threads, 0) : throw(ArgumentError("Barrier size must be > 0"))
end

"""
    sync(b::Barrier)

Stops all threads until they all reached the barrier and call `sync`.

See @Barrier
"""
function sync(b::Barrier)
    lock(b.notify)
    try
        b.cur_cnt += 1

        if b.cur_cnt == b.tar_cnt
            notify(b.notify, true)
            b.cur_cnt = 0
        else
            wait(b.notify)
        end
    finally
        unlock(b.notify)
    end
    return nothing
end
