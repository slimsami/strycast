# Strycast's Gufo extensions

Each directory here generates one external Gufo extension from the Strycast
tree. Today there is one: [`qwen35moe`](qwen35moe/README.md), published as
`strycast_qwen35moe`.

## Naming

The `strycast_` prefix states provenance: the extension is generated from
this tree and published by Strycast's CI, so its problems are reported here.
It does not mean "for Strix Halo", and it does not mean "endorsed by
Strycast".

- An extension generated here takes the `strycast_` prefix.
- An extension built and released from its own repository keeps its own name,
  even when a Strycast maintainer maintains it or it targets Strix Halo.
  Flint (`flint`) is one.
- An extension that moves into this tree and its CI is renamed then, as one
  coordinated change with its registry entry. Renaming a registered extension
  for any other reason would break existing installs.
