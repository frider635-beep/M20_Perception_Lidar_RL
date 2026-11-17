# V11-FD baseline ledger — 2026-09-02

Authority directory:
`logs/rsl_rl/M20-V11-Continuous-Landing/2026-08-27_18-14-26_v11_from_scratch`

SHA256:

- `params/env.yaml`: `b5b74d0a67eb0e11d403eb0e1692e16e355c8084ea23cd46a6fc59cc3602b099`
- `params/agent.yaml`: `a0fc5d884127206ed828f84b685fc30507fb233dd7e38e6c85d2d7dac0581224`
- `actor_observation_manifest.txt`: `656654578f7e519bd0124280155595ee32eb6f25d6f7466fac28d357b742a877`
- `model_20000.pt` comparison checkpoint: `8d4983978bbd04120e4c544efa5cc1d0e58024a5a81e52d66cdb09ba6d01a542`

V11-FD trains from scratch. `model_20000.pt` is retained only as the behavioral
comparison baseline and is not loaded into the new optimizer or networks.
