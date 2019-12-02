N = 4

hilb = HomogeneousSpin(N)
s = state(hilb)
vals = Int[]
for i=1:spacedimension(hilb)
    set!(s, hilb, i)
    push!(vals, toint(s, hilb))
end

@test all(vals .== 1:spacedimension(hilb))

shilb = SuperOpSpace(hilb)
@test physical(shilb) === hilb

s = state(shilb)
vals = Int[]
for i=1:spacedimension(shilb)
    set!(s, shilb, i)
    push!(vals, toint(s, shilb))
end
@test all(vals .== 1:spacedimension(shilb))
