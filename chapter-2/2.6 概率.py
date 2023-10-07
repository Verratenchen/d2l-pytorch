# 2.6.1
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fait_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(1, fait_probs).sample())

print(multinomial.Multinomial(10, fait_probs).sample())

counts = multinomial.Multinomial(1000, fait_probs).sample()
print(counts / 1000)

counts = multinomial.Multinomial(10, fait_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();

d2l.plt.show()


# 2.6.2


# 2.6.3
