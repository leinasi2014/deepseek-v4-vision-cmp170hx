# PP4 deployment

## Required inputs

- A native image built from one repository commit.
- `DeepSeek-V4-Flash-Vision-Exp` weights on a local filesystem.
- An API key stored in a separate, permission-restricted file.
- Four validated SM80 GPUs.

```bash
install -m 600 /dev/null /run/secrets/dsv4-api-key
printf '%s' 'replace-with-a-random-secret' \
  > /run/secrets/dsv4-api-key

export IMAGE=dsv4-cmp170hx:local
export MODEL_PATH=/models/DeepSeek-V4-Flash-Vision-Exp
export API_KEY_FILE=/run/secrets/dsv4-api-key
export GPU_DEVICES=0,1,2,3
export HOST_PORT=9016
bash deploy/serve-pp4.sh
```

The script launches the validated PP4 profile and waits for `/health`.

## Production cutover

For an existing service:

1. Record the current container ID, image ID, launch contract, and health.
2. Keep the old immutable container as a stopped, named rollback point.
3. Start the new image using the same model and API identity.
4. Require health, authenticated `/v1/models`, a real completion, 1M config
   evidence, and an empty fatal-log scan.
5. Automatically restore the old container on any failure.
6. Keep the rollback container until the new service has accumulated enough
   production traffic.

Do not delete the rollback image during the same operation that promotes a new
one.

## Driver and P2P gate

`nvidia-smi topo -m` is not a correctness test. After any driver/hardware
change, verify:

- CUDA peer access and peer reads/writes;
- NCCL transport (`P2P/CUMEM` or the expected peer path);
- multi-rank correctness under load;
- no Xid or CUDA fatal errors.

The driver/unlock procedure is intentionally outside this source repository.

## API security

The vLLM process receives the API key as a command-line option, so users with
permission to inspect the container may be able to read it. Keep Docker access
restricted and use a trusted reverse proxy for public exposure.

## 2026-09-03: sweep results and the bat2048 incident

`NCCL_P2P_LEVEL=SYS` is the validated headline win (+27% single-stream).
**`serve-pp4.sh` keeps `--max-num-batched-tokens 4096`**: `SYS + bat2048`
measured faster prefill/C16 on the test group (4236 prefill / 329.9 C16 vs
3525 / 330.8 baseline) but **crashed production with Xid 31 on the drafter
rank after 67 minutes of live traffic**; 4096 has 13h+ zero-Xid production
uptime. Do not raise `--gpu-memory-utilization` to 0.93 or change the
partition on top of SYS — both regress. Full incident and data:
[docs/cmp170hx/OPTIMIZATION-2026-09-03.md](../docs/cmp170hx/OPTIMIZATION-2026-09-03.md).
