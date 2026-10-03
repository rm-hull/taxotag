# Changelog

All notable changes to this project are documented here.

<!-- version list -->

## v0.1.3 (2026-10-03)

### Bug Fixes

- Use model's input dtype instead of hardcoded float32
  ([`1b7e0b9`](https://github.com/rm-hull/taxotag/commit/1b7e0b96b74a11882c6c3bb0d432ff39bb6bc835))

### Build System

- **deps**: Bump actions/download-artifact from 7 to 8
  ([#5](https://github.com/rm-hull/taxotag/pull/5),
  [`5e8bf17`](https://github.com/rm-hull/taxotag/commit/5e8bf17a786a91a9fb90b48a96336199d511b29a))

- **deps**: Bump actions/upload-artifact from 6 to 7
  ([#7](https://github.com/rm-hull/taxotag/pull/7),
  [`b12350a`](https://github.com/rm-hull/taxotag/commit/b12350a3afff446749ac63725b334e9e2c8fffff))

- **deps**: Bump astral-sh/setup-uv from 10.1.0 to 10.2.0
  ([#4](https://github.com/rm-hull/taxotag/pull/4),
  [`f83382b`](https://github.com/rm-hull/taxotag/commit/f83382b3bbd57eca93e9ba5a14a6d6f7ee3423b7))

- **deps**: Bump python-semantic-release/python-semantic-release
  ([#8](https://github.com/rm-hull/taxotag/pull/8),
  [`cae0917`](https://github.com/rm-hull/taxotag/commit/cae0917281e8cda547bde4d60b667a28d13dd256))

- **deps**: Update lockfile dependencies ([#10](https://github.com/rm-hull/taxotag/pull/10),
  [`17775f6`](https://github.com/rm-hull/taxotag/commit/17775f69826668ec52ac77deb6604aa9172c7114))

- **deps-dev**: Bump uv from 0.12.17 to 0.12.20 ([#9](https://github.com/rm-hull/taxotag/pull/9),
  [`96f830f`](https://github.com/rm-hull/taxotag/commit/96f830f1f5f1d3b53c20e5925bcd0ad4b96e8021))

- **deps-dev**: Bump uv from 0.9.17 to 0.12.17 ([#6](https://github.com/rm-hull/taxotag/pull/6),
  [`e41efed`](https://github.com/rm-hull/taxotag/commit/e41efed39efe3a1dccdc24f5a735a2d928e58709))

### Chores

- Conditionalize PR creation on model update
  ([`ec522a7`](https://github.com/rm-hull/taxotag/commit/ec522a7a2012680fdfbf0b03207b6d2028083b95))

- Update dependencies in uv.lock
  ([`baf8548`](https://github.com/rm-hull/taxotag/commit/baf854839bd8ed6e2f7f871cbef8975bf175e47d))


## v0.1.2 (2026-09-20)

### Bug Fixes

- Update model revision to v2.2.0 ([#3](https://github.com/rm-hull/taxotag/pull/3),
  [`b9d8ca6`](https://github.com/rm-hull/taxotag/commit/b9d8ca6029caa6e1b313a96a7011f292363ccf2c))

### Build System

- Add automated model revision updates
  ([`c2897fb`](https://github.com/rm-hull/taxotag/commit/c2897fb831d5516154811b4ea121ee250413e468))

- Update UV_VERSION in CI workflow
  ([`c0c9d22`](https://github.com/rm-hull/taxotag/commit/c0c9d229b778f9352596b15233256c5a093ee03f))

### Continuous Integration

- Configure automated semantic-release workflow
  ([`ef2af47`](https://github.com/rm-hull/taxotag/commit/ef2af476f142d1d29f262bee54017a98aaf60a82))

- Consolidate release jobs into CI workflow
  ([`e585d24`](https://github.com/rm-hull/taxotag/commit/e585d24bca8cbbdf0ca63ffba9da1ec1ab844455))

### Documentation

- Add AGENTS.md guide for AI assistants
  ([`f375e60`](https://github.com/rm-hull/taxotag/commit/f375e601b360351a85fa7a859d7c3d9ac0f4ac00))

- Add comprehensive library usage guide to README
  ([`749e266`](https://github.com/rm-hull/taxotag/commit/749e266fd55606fc3b38e989f385a319b5ec62d4))

- Link Hugging Face repo in README model asset description [skip ci]
  ([`afe77c0`](https://github.com/rm-hull/taxotag/commit/afe77c01e7532ed9cd9d52285d4a723c9d5f95b9))

### Testing

- Skip slow tests when runtime is missing
  ([`91e812f`](https://github.com/rm-hull/taxotag/commit/91e812ff437cebe71db9784d4bd8239a69901571))
