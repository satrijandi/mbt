# The deployable unit (DESIGN.md section 5 step 5): the EXACT runner image
# that compiled the manifest, plus the project source and that compiled
# manifest (target/manifest.json is in the build context - the bake step
# runs after the build step in the same Woodpecker workspace).
#
# Same-image bake is what makes ADR-19 env_digest verification hold by
# construction: `mbt run --manifest` inside this image sees the environment
# the manifest was compiled in, byte for byte.
FROM mbt-showcase-runner:dev

COPY . /app/project
WORKDIR /app/project
