# Mosaic workflow. `make all` rebuilds everything offline from the committed cache and runs;
# `make device` is the only target that needs network (Groq + a local llama.cpp server).
SHELL := /bin/bash
DEVICE_LIVE := device-C1-live,device-C2-live,device-C3-live,device-C6-live

.PHONY: install data device calibrate device-sim profile train eval report demo test cov live-test verify stats all

install:
	pip install -e '.[dev]'

data:                       ## corpora + question sets (deterministic, ~5 s)
	mosaic gen

device:                     ## LIVE device study: 12 questions at 8k on Groq C1/C2/C3 and the local CPU (C6)
	mosaic run --pipeline P1 --sizes 8k --n-per-class 4 --fixed C3 --planner C3 --live C3 --plans-only --yes
	for c in C1 C2 C3 C6; do \
	  mosaic run --pipeline P1 --sizes 8k --n-per-class 4 --fixed $$c --planner C3 --live $$c,C3 --run-id device-$$c-live --yes & \
	done; wait

calibrate:                  ## fit the simulator's error model from the live, oracle-labelled map calls (offline replay)
	mosaic calibrate --runs $(DEVICE_LIVE)

device-sim:                 ## the Cerebras rows of the device study (simulated: Cerebras unreachable from the build machine)
	for c in C4 C5; do \
	  mosaic run --pipeline P1 --sizes 8k --n-per-class 4 --fixed $$c --planner C3 --replay C3 --run-id device-$$c-sim --yes; \
	done

profile:                    ## latency profiles + routing labels on the profiling split
	mosaic profile --candidates C4,C5,C6 --from-runs $(DEVICE_LIVE)

train:
	mosaic train-router

eval:                       ## the four pipelines on all 720 evaluation questions (simulated main pool, virtual time)
	for p in P0 P1 P2; do mosaic run --pipeline $$p --run-id main-$$p --progress-every 120 || exit 1; done
	mosaic run --pipeline P3 --run-id main-P3 --explore 0 --progress-every 120
	mosaic run --pipeline P3 --run-id main-P3x --explore 0.05 --progress-every 120
	mosaic eval --a main-P0 --b main-P3 || true

report:
	mosaic report

demo:
	mosaic demo --from-cache --speed 4x

test:
	pytest -q

cov:
	pytest -q --cov=mosaic --cov-report=term --cov-report=json:report/coverage.json

live-test:
	pytest -m live -q

stats:                      ## core line count + coverage summary for the report (M7)
	python scripts/code_stats.py

verify:                     ## M6: cached re-run reproduces the runs and the report byte for byte
	python scripts/verify_repro.py

all: data calibrate device-sim profile train eval cov stats verify report
