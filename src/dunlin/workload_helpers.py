"""Stable prompt and static-batch construction shared by inference workloads."""

from __future__ import annotations

from transformers import BatchEncoding

DEFAULT_INPUT_TOKENS = 500

DEFAULT_PROMPT = """Explain how a modern city might redesign its transportation system over the next twenty years to reduce congestion, energy use, pollution, and travel time while still remaining affordable and convenient for residents.

Consider a hypothetical city with approximately three million residents. The city currently relies heavily on private cars, although it also has buses, commuter rail, a subway system, bicycle lanes, and pedestrian infrastructure. Population growth is expected to continue, and several new residential and commercial districts are being developed outside the traditional city center.

Discuss how transportation planners should think about the relationship between public transit, private vehicles, walking, cycling, and emerging technologies. Explain which investments are likely to provide the greatest long-term benefits and which approaches might appear attractive but create unexpected problems.

Include discussion of rail capacity, bus networks, dedicated bus lanes, congestion pricing, parking policy, electric vehicles, autonomous vehicles, bicycle infrastructure, pedestrian-friendly neighborhoods, and land-use planning. Consider how the design of housing and employment centers affects transportation demand. Explain why transportation systems should not be evaluated only by vehicle speed, but also by accessibility, reliability, capacity, cost, safety, and environmental impact.

Describe several important tradeoffs. For example, adding road capacity can initially reduce congestion but may encourage additional driving. Electric vehicles can reduce local emissions but still require road space and parking. High-capacity rail can move large numbers of people efficiently but requires substantial capital investment. Bus systems are often cheaper and more flexible, but they may become unreliable when buses operate in mixed traffic. Bicycle infrastructure can be inexpensive relative to major rail projects, but its usefulness depends on network connectivity, safety, weather, geography, and local travel patterns.

Also consider the role of technology. Real-time transit information, integrated payment systems, adaptive traffic signals, demand-responsive buses, and better routing software may improve existing infrastructure without requiring entirely new transportation networks. At the same time, technology should not be treated as a substitute for basic physical capacity or sound urban planning.

Finally, propose a phased strategy for the city. Identify actions that could be implemented within five years, investments that might require ten years, and structural changes that could take twenty years. Explain how planners could measure whether the strategy is working. Use quantitative metrics where appropriate, such as average travel time, passenger throughput, operating cost, emissions per trip, reliability, and the percentage of residents who can reach major employment centers within a reasonable amount of time."""
# Repeating the source supplies a long prompt. Left truncation preserves the
# same trailing token sequence used by the recorded workloads.
BENCHMARK_PROMPT = "\n\n".join([DEFAULT_PROMPT] * 9)


def repeat_batch(inputs: BatchEncoding, batch_size: int) -> BatchEncoding:
    """Repeat one tokenized request into an identical static batch."""
    return BatchEncoding(
        {
            name: tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1))
            for name, tensor in inputs.items()
        }
    )
