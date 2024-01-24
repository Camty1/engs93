bolts = 20
not_in_spec = 5
p_caught = 0.95


def calc_prob(
    events: str, bolts: int, not_in_spec: int, p_caught: float
) -> float:
    p = 1
    for event in events:
        if event == "0":
            p *= not_in_spec / bolts * (1 - p_caught)
            not_in_spec -= 1
        else:
            p *= (bolts - not_in_spec) / bolts

        bolts -= 1

    return p


def decimal2bin(num: int) -> str:
    return '{:04b}'.format(num)


total = 0
for i in range(16):
    events = decimal2bin(i)
    total += calc_prob(events, bolts, not_in_spec, p_caught)

print(total)
print(1-total)
