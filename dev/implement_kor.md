# optimalcontrol로 Bruker Pulse 만들기

이 저장소에서 pulse를 만드는 방법은 크게 두 가지입니다.

1. 해석해를 바로 샘플링: `optimalcontrol.rope.rope_waveform()`, `optimalcontrol.crop.crop_waveform()`
2. 목표 전달에 맞춰 수치 최적화: `optimalcontrol.grape.ControlProblem` + GRAPE

실제로 Bruker에서 바로 써 볼 `.shape` 파일이 필요하면, 이 문서에서는 [`examples/sciadv2023_fig1_ur180.py`](examples/sciadv2023_fig1_ur180.py)를 메인 예제로 보는 것이 가장 좋습니다. 이 예제는 phase-only GRAPE로 low-power UR-180 pulse를 만들고, Bruker shape와 진단 그림까지 같이 생성합니다.

## 준비

이 문서의 예제는 저장소 루트에서 실제로 실행 검증했습니다.

- Python `3.13.13`
- SciPy `1.17.1`
- Matplotlib `3.10.9`

예제 스크립트는 패키지 import를 쓰므로 아래처럼 맞추면 됩니다.

```bash
python3 -m pip install scipy
python3 -m pip install matplotlib
python3 -m pip install -e .
```

개발 환경 전체를 맞추려면 아래 한 줄이면 충분합니다.

```bash
pip install -e ".[dev]"
```

## 메인 예제: `sciadv2023` Bruker shape 생성

가장 빠른 실행 방법은 아래입니다.

```bash
python3 -m examples.sciadv2023_fig1_ur180
```

실제 실행 결과는 아래와 같았습니다.

```text
Saved Bruker shape /home2/dlee/project/optimalcontrol/examples/output/sciadv2023_fig1_ur180.shape
Saved figure /home2/dlee/project/optimalcontrol/examples/output/sciadv2023_fig1_ur180.png
```

생성 파일:

- `examples/output/sciadv2023_fig1_ur180.shape`
- `examples/output/sciadv2023_fig1_ur180.png`

이 예제의 pulse 조건:

- `N_STEPS = 72`
- 총 길이 `540 us`
- step 길이 `7.5 us`
- nominal RF `7.5 kHz`
- offset 대역 `+/-6.3 kHz`
- B1 보상 범위 `+/-15 %`

즉, 이 예제는 "상수 amplitude + 시간에 따라 변하는 phase" 형태의 UR-180 pulse를 Bruker shape로 내보내는 샘플입니다.

## Bruker shape 내용

실제로 생성된 `.shape` 파일 헤더는 아래처럼 들어갑니다.

```text
##TITLE= sciadv2023_fig1_ur180
##$SHAPE_TOTROT= 1.800000e+02
##$SHAPE_INTEGFAC= 1.000000e+00
##$SHAPE_MODE= 1
##$OPTIMALCONTROL_TOTAL_DURATION_S= 5.400000000000e-04
##$OPTIMALCONTROL_STEP_DURATION_S= 7.500000000000e-06
##$OPTIMALCONTROL_RF_HZ= 7.500000000000e+03
##$OPTIMALCONTROL_BANDWIDTH_HZ= 6.300000000000e+03
##$OPTIMALCONTROL_B1_DEVIATION_PERCENT= 1.500000000000e+01
##$OPTIMALCONTROL_NOTE= Set pulse length to TOTAL_DURATION_S and calibrate 100% to RF_HZ.
##NPOINTS= 72
##XYPOINTS= (XY..XY)
1.000000000e+02, 6.598765774e+00
1.000000000e+02, 1.723134983e+01
...
```

읽는 방법:

- 첫 번째 열은 amplitude `%`
- 두 번째 열은 phase `degree`
- 이 예제는 constant-amplitude pulse라서 amplitude가 전 구간 `100`입니다

Bruker 쪽에서 맞춰야 할 핵심 값:

- pulse length: `540 us`
- amplitude `100%`가 실제 `7.5 kHz`가 되도록 RF calibration

문서상으로는 `.shape`만 있으면 끝이 아니라, 위 두 값이 장비 설정과 같이 맞아야 pulse가 의도대로 동작합니다.

## Python에서 직접 호출

스크립트를 subprocess로 실행하지 않고 Python에서 바로 호출해도 됩니다.

```python
from examples.sciadv2023_fig1_ur180 import run

result = run(optimize=False)
phase_deg = result[:72]
summary = result[-6:]
```

`result`는 길이 `78`의 배열입니다.

- 앞 `72`개는 각 time slice의 phase 값입니다. 단위는 degree이고 `0..360`으로 wrap되어 있습니다.
- 뒤 `6`개는 성능 요약입니다. 순서는 평균 전달 효율, 전달 효율 표준편차, 평균 `Mxy`, `Mxy` 표준편차, 평균 phase error, phase error 표준편차입니다.

실제 검증 시 마지막 6개 값은 아래였습니다.

```text
[0.997738, 0.00322, 0.999055, 0.001462, 0.090669, 2.942966]
```

## 다시 최적화하고 싶을 때

기본 실행은 저장소에 들어 있는 cached phase 결과를 사용합니다. phase-only GRAPE를 다시 돌려 shape를 새로 만들고 싶으면 아래처럼 실행합니다.

```bash
python3 -m examples.sciadv2023_fig1_ur180 --optimize
```

이 경로는 시간이 더 걸리지만, 현재 코드 기준으로 pulse를 다시 재생성하는 표준 방법입니다.

## 구조를 이해하기 위한 최소 GRAPE 예제

`sciadv2023` 예제가 실제 shape 생성에는 가장 유용하지만, 내부 구조를 이해하려면 더 작은 `ControlProblem` 예제가 편합니다. 아래 예제는 단일 spin에서 `Iz -> Ix` 전달을 만드는 가장 작은 형태의 XY pulse 최적화입니다.

```python
from pathlib import Path

import numpy as np

from optimalcontrol.grape import ControlProblem, grape_xy
from optimalcontrol.io import export_bruker, export_csv, export_json
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.optimizers import run_grape
from optimalcontrol.states import normalise_hs

output_dir = Path("examples/output/minimal_grape_pulse")
output_dir.mkdir(parents=True, exist_ok=True)

cp = ControlProblem(
    drifts=[np.zeros((2, 2), dtype=np.complex128)],
    operators=[np.complex128(-1j) * Ix(), np.complex128(-1j) * Iy()],
    rho_init=[normalise_hs(Iz())],
    rho_targ=[normalise_hs(Ix())],
    pulse_dt=0.1,
    pwr_levels=[1.0, 1.0],
    freeze=None,
    fidelity_mode="real",
    basis="dense",
)

wfm0 = np.zeros((4, 2), dtype=np.float64)
initial_fidelity = grape_xy(cp, wfm0)
waveform, result = run_grape(cp, wfm0, method="lbfgs", m=4, tol_x=0.0, tol_g=0.0, max_iter=20)

export_json(waveform, output_dir / "pulse.json")
export_csv(waveform, output_dir / "pulse.csv")
export_bruker(waveform, output_dir / "pulse.shape")

print(f"initial fidelity: {initial_fidelity:.6f}")
print(f"final fidelity:   {result.fidelity_final:.6f}")
print(f"iterations:       {result.n_iter}")
print(f"reason:           {result.reason}")
print("internal shape:", result.wfm_final.shape)
print("export shape:  ", waveform.data.shape)
print("final waveform:")
print(result.wfm_final)
```

실제 실행 결과:

```text
initial fidelity: 0.000000
final fidelity:   1.000000
iterations:       6
reason:           step_tol
internal shape: (4, 2)
export shape:   (2, 4)
final waveform:
[[0.         3.92699082]
 [0.         3.92699082]
 [0.         3.92699082]
 [0.         3.92699082]]
```

이 예제에서 봐야 할 점:

- 내부 waveform 형식은 `(n_steps, n_channels)`
- export된 `Waveform.data`는 `(n_channels, n_steps)`
- `run_grape()`는 최적화 결과와 export용 waveform을 같이 반환

단, 여기서 쓰는 `export_bruker()`는 최소 호환용 stub입니다. 실제 장비용 shape 예제로는 여전히 `sciadv2023_fig1_ur180.py` 쪽이 더 현실적입니다.

## 더 큰 문제로 확장하는 방법

`sciadv2023` 예제나 최소 예제에서 보통 아래 항목만 바꾸면 됩니다.

1. `drifts`: 화학적 shift, J-coupling, relaxation 반영
2. `operators`: `Ix`, `Iy`, `Sx`, `Sy` 등 실제 제어 채널로 확장
3. `rho_init`, `rho_targ`: 원하는 전달 예를 정의, 예: `Iz -> 2IzSz`
4. `n_steps`, `pulse_dt`: pulse 길이와 해상도 조절
5. `offsets`, `offset_operators`, `pwr_levels`: broadband, offset robustness, B1 robustness 추가

더 복잡한 GRAPE 예제는 [`examples/grape_broadband_180.py`](examples/grape_broadband_180.py), [`examples/jmr2005_fig5_rope.py`](examples/jmr2005_fig5_rope.py), [`tests/test_integration.py`](tests/test_integration.py)를 보면 바로 이어집니다.

## 해석해 pulse가 필요한 경우

최적화 대신 바로 파형을 샘플링하고 싶으면 아래 API를 쓰면 됩니다.

```python
from optimalcontrol.crop import crop_pulse_params, crop_waveform
from optimalcontrol.rope import rope_waveform

rope = rope_waveform(T=0.263 / 100.0, n=1.0, J_hz=100.0, dt=(0.263 / 100.0) / 400.0)
ka = 0.6 * 100.0
kc = 0.75 * ka
params = crop_pulse_params(ka, kc, J_hz=100.0)
crop = crop_waveform(ka, kc, J_hz=100.0, dt=1e-4, truncation_window=params.truncation_window)
```

이 경로는 "이론식이 이미 정해진 pulse를 샘플링"하는 방식이고, GRAPE는 "목표 전달에 맞춰 pulse를 새로 찾는" 방식입니다.
