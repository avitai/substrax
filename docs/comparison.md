# How substrax compares

substrax is a substrate, not a trainer. It owns one implementation of each concern a
JAX project needs before it can train anything: process configuration, device
discovery and sharding, checkpointing, stopping rules, run logging, and job
submission to remote compute. It does not own a training loop, a model zoo, or an
opinion about what you are training.

That is the axis every comparison below turns on. kauldron, axlearn, tunix and
MaxText are assemblies: each composes roughly the same dependencies substrax wraps
and adds a loop, a config system and a set of models. Choosing between them is
choosing whose training loop to adopt. Choosing substrax is choosing not to write
the layer underneath yours.

All versions and pins on this page were measured on 2026-10-02 against each
project's default branch.

## At a glance

| | substrax | kauldron | axlearn | tunix | MaxText |
| --- | --- | --- | --- | --- | --- |
| What it is | the layer under a trainer | a research library for quick iteration | large-model training on heterogeneous infrastructure | LLM post-training | a reference LLM implementation |
| Owns a training loop | no | yes | yes | yes | yes |
| Owns models | no | no | yes | yes | yes |
| Scope | one home per concern | config-driven experiments | end-to-end large-model stack | SFT, DPO, PPO, GRPO, GSPO, agentic RL | decoder-only LLMs at scale |
| Depends on an Avitai package | none | n/a | n/a | n/a | n/a |

## The dependency base is nearly the same

This is what makes the comparison meaningful rather than a feature table. substrax
and its closest comparators sit on the same four or five packages:

| | jax | flax | optax | orbax-checkpoint | grain |
| --- | --- | --- | --- | --- | --- |
| substrax | `>=0.11.1,<0.11.2` | `>=0.12.9` | `>=0.2.8` | `>=0.11.33` | no |
| kauldron | unpinned | yes | yes | yes | no |
| tunix | `>=0.6.0,!=0.7.2,<0.11.1` | `>=0.12.5` | yes | `>=0.12.0` | yes |
| axlearn | `==0.8.3` | no | no | no | no |
| MaxText | unpinned | yes | yes | yes | no |

Two things follow from that table, and both matter more than any feature list.

**The overlap is the argument.** tunix composes flax, jax, optax, orbax and grain
and assembles them into trainers. substrax composes the same packages and assembles
them into nothing: it hands you `substrax.mesh`, `substrax.checkpoint`,
`substrax.spmd` and leaves the loop to you. If you are writing your own training
code, that difference is the whole decision.

**The pins are disjoint, so this is a positioning page and not a compatibility
promise.** tunix caps jax below `0.11.1`, which is exactly substrax's floor, so the
two cannot be installed together today. axlearn pins `jax==0.8.3` exactly, three
minor versions back. Read the comparisons below as architecture, never as "you can
mix these".

## substrax and kauldron

kauldron is a research library optimised for quick iteration and modularity, built
around a config system where an experiment is a declarative object. It owns the
loop, the evaluation harness and the experiment structure.

Reach for kauldron when you want the experiment framework to be given. Reach for
substrax when you already have a loop, or when the loop is the part you want to
own, and what you need is for sharding, checkpoint restore and run logging to have
one implementation rather than one per project.

They are not mutually exclusive in principle. In practice kauldron already answers
the concerns substrax owns, so using both means picking which answer wins.

## substrax and axlearn

axlearn is Apple's stack for training large models on heterogeneous
infrastructure. It materialises a full JAX program: mesh shape for the target
accelerator, sharding annotations, XLA auto-tuning, attention-kernel selection and
rematerialisation policy, with checkpointing, monitoring and fault tolerance in the
runtime.

It overlaps substrax most directly on `substrax.mesh`, `substrax.spmd` and
`substrax.checkpoint`, and it goes much further: axlearn decides the sharding for
you from the model structure, where substrax gives you a sharding vocabulary and
expects you to choose. The exact `jax==0.8.3` pin is the practical difference.
axlearn is a whole environment; substrax is a dependency.

## substrax and tunix

tunix is the sharpest contrast on this page, because it is the clearest example of
the thing substrax deliberately is not. It composes substrax's own dependency base
one layer up, and turns it into task-specific trainers: supervised fine-tuning,
preference tuning, knowledge distillation, PPO, GRPO and GSPO, plus agentic RL with
async vLLM and SGLang rollout.

Same foundation, opposite abstraction. tunix answers "how do I post-train an LLM".
substrax answers "where does checkpoint restore live so I only write it once".

## substrax and MaxText

MaxText is a reference implementation of decoder-only LLMs in JAX, written to be
read and forked as much as imported. Its value is a known-good, high-performance
training path end to end.

That is a different kind of artifact from either a library or a substrate. If a
MaxText model is what you want to train, start there. substrax is relevant to the
code you write around it, not to the model.

## When substrax is the wrong choice

Stated plainly, because a comparison page that never says no is marketing.

- **You want a training loop.** substrax does not have one and will not grow one.
  kauldron or tunix will serve you better.
- **You want a model.** substrax ships none. MaxText, axlearn and tunix all do.
- **You are on a jax older than 0.11.1.** The pin is deliberate and narrow, with the
  reason recorded inline in `pyproject.toml`, and `tests/ci/test_jax_cap.py` detects
  when the cap can lift. Until then, substrax is not installable alongside tunix or
  axlearn.
- **You have one project.** The case for substrax is that each concern gets one
  implementation and one test suite across several projects. With one project, the
  layer is overhead.

## When it is the right one

- Several JAX projects that keep re-implementing device detection, mesh
  construction, checkpoint save and restore, early stopping and run logging.
- A project that wants its training loop to stay its own.
- A team that wants the sharding and checkpoint behaviour under test somewhere
  other than inside each model repository.

Ten repositories currently pin substrax, and substrax pins none of them. That
asymmetry is the design: it is the bottom of the chain, so it can be a dependency
of anything without dragging the rest of a stack along.
