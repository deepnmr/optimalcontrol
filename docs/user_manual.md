# optimalcontrol 사용자 매뉴얼 (완전 초보자용 Step-by-Step 가이드)

이 문서는 NMR도, 파이썬 패키징도 처음인 사람이 **위에서 아래로 그대로 따라 하면**
첫 최적 제어 펄스를 만들고 분광계용 파일로 내보낼 수 있게 만든 안내서입니다.
각 단계마다 "복사해서 실행할 명령/코드"와 "무엇이 보이면 성공인지"를 함께 적었습니다.

> 이 매뉴얼에 나오는 모든 코드 조각은 실제로 실행해 통과를 확인한 것들입니다.
> 더 깊은 이론·API 설명은 `docs/` 폴더의 다른 `guide_*.md` 문서를 참고하세요.

---

## 목차

0. [이 패키지가 하는 일 (한 문단)](#0-이-패키지가-하는-일)
1. [설치하기](#1-설치하기)
2. [꼭 알아야 할 개념 6개](#2-꼭-알아야-할-개념-6개)
3. [5분 만에 첫 펄스 만들기 (GRAPE)](#3-5분-만에-첫-펄스-만들기-grape)
4. [결과 읽는 법](#4-결과-읽는-법)
5. [파일로 내보내기 (CSV / JSON / Bruker)](#5-파일로-내보내기)
6. [번들 예제 실행하기](#6-번들-예제-실행하기)
7. [해석적 펄스 (ROPE / CROP) 빠른 레시피](#7-해석적-펄스-rope--crop)
8. [강건한 펄스 만들기 (오프셋 / B1 앙상블)](#8-강건한-펄스-만들기)
9. [단위 치트시트 (가장 흔한 실수)](#9-단위-치트시트)
10. [문제 해결 (Troubleshooting)](#10-문제-해결)
11. [다음에 볼 문서](#11-다음에-볼-문서)

---

## 0. 이 패키지가 하는 일

`optimalcontrol`은 NMR 스핀에 걸어 줄 **RF 펄스 파형**을 설계하는 파이썬 패키지입니다.
"이 스핀 상태(`rho_init`)를 저 상태(`rho_targ`)로 바꾸고 싶다"고 말하면,
그 변환을 가장 잘 해내는 시간에 따른 RF 진폭·위상 파형을 찾아 줍니다.

세 가지 방법을 제공합니다.

| 방법 | 성격 | 언제 쓰나 |
|------|------|-----------|
| **GRAPE** | 수치 최적화 | 임의의 목표를 유연하게. 대부분 여기서 시작 |
| **ROPE** | 해석적 공식 | 이완(relaxation) 하 헤테로핵 전이 |
| **CROP** | 해석적 공식 | 교차상관 이완 최적화 펄스 |

무거운 계산(전파, 그래디언트)은 병렬 Rust 확장에서 돌아가고, Rust가 없으면
자동으로 NumPy/SciPy로 대체됩니다. 여러분은 파이썬만 쓰면 됩니다.

---

## 1. 설치하기

### 1-1. 준비물 확인

터미널에서 아래를 실행해 파이썬 3.10 이상인지 확인합니다.

```bash
python3 --version
```

`Python 3.10.x` 이상이 보이면 됩니다.

미리 빌드된 wheel이 없는 환경이라면 소스에서 Rust 확장을 컴파일해야 하므로
Rust 툴체인도 필요합니다. macOS(Homebrew) 기준:

```bash
brew install rust
```

`rustc --version`이 버전을 출력하면 준비 완료입니다.

### 1-2. 가상환경 만들고 설치

프로젝트 폴더에서:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # 개발용 도구까지 함께 설치
```

일반 사용자는 PyPI에서 바로 설치할 수도 있습니다.

```bash
pip install optimalcontrol
```

### 1-3. 설치 확인

아래 한 줄이 버전 문자열을 출력하면 설치 성공입니다.

```bash
python3 -c "import optimalcontrol; print(optimalcontrol.__version__)"
```

예상 출력:

```
0.3.0
```

전체 테스트로 확실히 확인하고 싶다면:

```bash
python3 -m pytest -q
```

`203 passed` 같은 줄이 보이면 환경이 완전히 정상입니다.

---

## 2. 꼭 알아야 할 개념 6개

코드를 쓰기 전에 딱 6개 용어만 이해하면 됩니다.

1. **`ControlProblem`** — "무엇을 풀지"를 담는 상자.
   드리프트/제어 연산자, 시작·목표 상태, 시간 간격, RF 세기 등을 한데 모읍니다.

2. **파형(waveform)의 모양** — 항상 `(n_steps, n_channels)` 형태의 2차원 배열입니다.
   **행 = 시간 조각**, **열 = 제어 채널**(보통 x, y 두 채널). 헷갈리면 이 문장만 기억하세요.

3. **생성자(generator)** — 시간 전개를 만드는 연산자. GRAPE의 `dense` 경로에서는
   반-에르미트(anti-Hermitian) 형태 `-1j * Ix()` 처럼 넣습니다. `1j`를 빼먹는 것이 초보자
   최다 실수입니다.

4. **상태(state)** — 벡터 또는 정사각 밀도행렬. 밀도행렬은 `normalise_hs(...)`로
   Hilbert–Schmidt 정규화해서 넣습니다.

5. **`fidelity_mode`** — 목표 달성도를 재는 방식. `"real"`, `"imag"`, `"abs2"` 중 하나.
   전역 위상이 상관없으면 `"abs2"`, 부호까지 맞춰야 하면 `"real"`.

6. **`pwr_levels`와 `pulse_dt`** — 각 채널의 RF 세기와 한 시간 조각의 길이.
   길이(`operators` 개수)와 `pwr_levels` 개수는 반드시 같아야 합니다.

이 여섯 개면 첫 펄스를 만들 수 있습니다.

---

## 3. 5분 만에 첫 펄스 만들기 (GRAPE)

목표: 한 스핀을 `Iz` 상태에서 `Ix` 상태로 옮기는 펄스(90도 회전에 해당)를 찾습니다.

`my_first_pulse.py` 라는 파일을 만들고 아래를 그대로 붙여 넣으세요.

```python
import numpy as np

import optimalcontrol
from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.optimizers import run_grape
from optimalcontrol.states import normalise_hs

# (1) 재현 가능하게 난수 시드 고정
optimalcontrol.set_random_seed(0)

# (2) 풀 문제를 상자에 담기
cp = ControlProblem(
    drifts=[np.zeros((2, 2), dtype=np.complex128)],          # 자유 세차 없음(온-레조넌스)
    operators=[np.complex128(-1j) * Ix(),                    # x 채널 제어
               np.complex128(-1j) * Iy()],                   # y 채널 제어
    rho_init=[normalise_hs(Iz())],                           # 시작: Iz
    rho_targ=[normalise_hs(Ix())],                           # 목표: Ix
    pulse_dt=0.1,                                            # 한 조각 길이(무차원 예제)
    pwr_levels=[1.0, 1.0],                                   # 채널별 RF 세기
    freeze=None,
    fidelity_mode="real",
    basis="dense",
)

# (3) 초기 추정 파형: 모두 0에서 시작 (8 스텝 x 2 채널)
wfm0 = np.zeros((8, 2), dtype=np.float64)

# (4) 최적화 실행
waveform, result = run_grape(cp, wfm0, method="lbfgs", max_iter=100)

# (5) 결과 보기
print("수렴했나요? :", result.converged)
print("최종 fidelity :", round(result.fidelity_final, 6))
print("반복 횟수     :", result.n_iter)
print("파형 채널     :", waveform.channels)
print("파형 데이터 shape :", waveform.data.shape)
```

실행:

```bash
python3 my_first_pulse.py
```

예상 출력(값은 환경에 따라 미세하게 다를 수 있음):

```
수렴했나요? : True
최종 fidelity : 1.0
반복 횟수     : 5
파형 채널     : ['x', 'y']
파형 데이터 shape : (2, 8)
```

**`최종 fidelity`가 1.0에 가깝다면 성공**입니다. 방금 여러분은 Iz → Ix 전이 펄스를 찾았습니다.

> 헷갈리는 점: 입력 `wfm0`는 `(n_steps, n_channels) = (8, 2)`인데,
> 내보내진 `waveform.data`는 `(n_channels, n_steps) = (2, 8)`입니다.
> **최적화 입력은 시간이 행, 내보내기 파형은 시간이 열**입니다. 5번 개념 참고.

---

## 4. 결과 읽는 법

`run_grape`는 두 개를 돌려줍니다: `waveform`(내보낼 수 있는 파형)과 `result`(최적화 리포트).

`result`(= `OptimResult`)의 주요 필드:

| 필드 | 뜻 |
|------|-----|
| `fidelity_final` | 최종 달성도(1.0에 가까울수록 좋음) |
| `converged` | 수렴 조건 충족 여부 |
| `n_iter` | 반복 횟수 |
| `n_feval` | 목적함수 평가 횟수 |
| `reason` | 최적화가 멈춘 이유 문자열 |
| `history` | 반복별 fidelity 기록 (리스트) |
| `wfm_final` | 최종 파형 `(n_steps, n_channels)` |

`waveform`(= `Waveform`)의 주요 필드:

| 필드 | 뜻 |
|------|-----|
| `channels` | 채널 이름 리스트, 예: `['x', 'y']` |
| `data` | `(n_channels, n_steps)` 실수 배열 |
| `times` | 각 시간 조각의 시각 |
| `units` | 단위 문자열 |
| `metadata` | 부가 정보 |
| `problem_hash` | 이 파형을 만든 문제의 해시(재현성 추적용) |

수렴 곡선을 보고 싶으면:

```python
print(result.history)      # [f0, f1, ...] fidelity가 올라가는 과정
```

---

## 5. 파일로 내보내기

찾은 파형을 세 가지 형식으로 저장할 수 있습니다.

```python
from optimalcontrol.io import export_csv, export_json, export_bruker

export_csv(waveform, "pulse.csv")        # 사람이 읽는 표 형식
export_json(waveform, "pulse.json")      # 메타데이터까지 온전히 보존
export_bruker(waveform, "pulse.bruker")  # Bruker 분광계용 진폭/위상(deg) 형식
```

- **CSV**: 시간·채널 값을 표로. 빠르게 눈으로 확인할 때.
- **JSON**: 채널, 단위, 메타데이터, 문제 해시까지 저장. 나중에 `import_json`으로 그대로 복원.
- **Bruker**: x/y 두 채널을 진폭과 위상(도 단위)으로 변환해 분광계에 바로 로드.

다시 불러오기:

```python
from optimalcontrol.io import import_json
same_waveform = import_json("pulse.json")
```

---

## 6. 번들 예제 실행하기

패키지에는 논문을 재현하는 예제가 `examples/` 폴더에 들어 있습니다.
**직접 파형을 짜기 전에 예제 하나를 돌려 보면** 전체 흐름이 한눈에 들어옵니다.

교과서적인 REBURP 180도 밴드선택 펄스 참조 그림:

```bash
python3 -m examples.reburp_pulse
```

전체 GRAPE로 광대역 180도 펄스를 최적화하는 예제(캐시된 해를 사용, 즉시 실행):

```bash
python3 -m examples.grape_broadband_180
```

새로 최적화부터 다시 돌리려면 `--optimize`를 붙입니다:

```bash
python3 -m examples.grape_broadband_180 --optimize
```

출력물(그림, `.shape` 파일)은 `examples/output/` 에 저장됩니다.

사용 가능한 예제 목록 보기:

```bash
ls examples/*.py
```

---

## 7. 해석적 펄스 (ROPE / CROP)

최적화를 돌리지 않고 공식으로 바로 파형을 얻는 경로입니다.

**ROPE** — 유한 시간 제어와 RF 파형 샘플링:

```python
from optimalcontrol.rope import rope_g, rope_waveform

print(rope_g(2.0))   # ROPE 이득 계수

wf = rope_waveform(T=5e-3, n=2.0, J_hz=140.0, dt=1e-4)
print(sorted(wf.keys()))     # ['amplitude', 'phase', 'times', 'u1', 'u2']
print(len(wf["times"]))      # 시간 조각 개수
```

반환값은 딕셔너리입니다: `times`(초), `u1`/`u2`(무차원 제어), `amplitude`(rad/s), `phase`(라디안).

**CROP** — 대칭 절단 펄스 파형:

```python
from optimalcontrol.crop import crop_waveform

wf = crop_waveform(ka=0.0, kc=0.0, J_hz=140.0, dt=1e-4, truncation_window=5e-3)
print(sorted(wf.keys()))     # ['amplitude', 'irrad_freq', 'times']
```

이론적 배경은 `docs/guide_rope_crop.md`를 참고하세요.

---

## 8. 강건한 펄스 만들기

실제 분광계에서는 오프셋(공명 어긋남)과 B1(RF 세기) 불균일이 있습니다.
`ControlProblem`에 앙상블 축을 추가하면 여러 조건에서 동시에 잘 동작하는 펄스를 찾습니다.

- **오프셋 강건성**: `offsets`(Hz 리스트)와 `offset_operators`(각 축의 연산자)를 지정.
- **B1 강건성**: `pwr_levels`에 여러 세기를 주어 RF 앙상블 축을 만듦.
- **위상 사이클**: `phase_cycle`로 위상 순환을 평균.

여러 드리프트/오프셋/B1을 조합하면 카테시안 곱 앙상블로 확장되어 모든 조합에 대해
평균 fidelity가 최적화됩니다. 자세한 사용법과 penalty(진폭 제한 등)는
`docs/guide_ensembles_penalties.md`에 정리되어 있습니다.

오프셋 프로파일을 그려 성능을 확인할 때는 Bloch 앙상블 전파를 씁니다:

```python
from optimalcontrol.bloch import propagate_bloch_ensemble
```

(구체적 인자는 `examples/grape_broadband_180.py`가 실제로 쓰는 방식을 그대로 참고하세요.)

---

## 9. 단위 치트시트

초보자가 가장 자주 막히는 부분이 **단위**입니다. 규칙은 다음과 같습니다.

| 대상 | 단위 |
|------|------|
| 공개 커플링 상수(J), 화학 이동 | **Hz** |
| `RelaxationRates`의 이완 항 | **rad/s** |
| `rope_waveform`의 `amplitude` | **rad/s** (각진폭) |
| `rope_waveform`의 `phase` | **라디안** |
| 3절 무차원 예제의 `pulse_dt`, `pwr_levels` | **무차원** (교육용) |
| 실제 실험용 파형 시간 | **초(s)** |

Hz를 각주파수로 바꿀 때는 항상 `2 * np.pi`를 곱합니다. 예: `2 * np.pi * 140.0`(140 Hz 커플링).
`-1j`를 빼먹거나 `2*pi`를 빼먹는 것이 잘못된 결과의 대표 원인입니다.

---

## 10. 문제 해결

**증상: fidelity가 낮은 값에서 즉시 멈춤 (`n_iter`가 매우 작음)**
- 생성자를 `-1j * Ix()`처럼 반-에르미트로 넣었는지 확인하세요. `1j`를 빼면 그래디언트가
  엉켜 초반에 정체됩니다.
- 상태를 `normalise_hs(...)`로 정규화했는지 확인하세요.
- 펄스가 목표에 도달할 만큼 충분한지: `pulse_dt`, 스텝 수, `pwr_levels`를 키워 보세요.

**증상: `pwr_levels length ... must match operator count` 오류**
- `operators`의 개수와 `pwr_levels`의 개수를 같게 맞추세요.

**증상: `waveform must have shape (n_steps, n_channels)` 오류**
- 최적화 입력 파형은 **행이 시간, 열이 채널**입니다. `(스텝, 채널)` 순서를 확인하세요.

**Rust 없이 순수 NumPy로 비교 실행하고 싶을 때**
```bash
OPTIMALCONTROL_DISABLE_RUST=1 python3 my_first_pulse.py
```

**린트·타입 검사**
```bash
ruff check .
mypy optimalcontrol
```

**막히면 참고할 것**: `examples/`의 실제 동작 코드가 가장 믿을 만한 사용 예시입니다.

---

## 11. 다음에 볼 문서

- `docs/guide_spin_system.md` — 스핀 시스템 구성
- `docs/guide_states.md` — 상태와 fidelity
- `docs/guide_operators.md` — 연산자 도구
- `docs/guide_grape.md` — GRAPE 심화(Hilbert/Liouville 경로)
- `docs/guide_rope_crop.md` — ROPE/CROP 이론
- `docs/guide_ensembles_penalties.md` — 앙상블과 penalty
- `docs/guide_paper_figures.md` — 논문 그림 재현
- `docs/spinach_mapping.md` — Spinach API와의 대응 관계

여기까지 따라왔다면 여러분은 이미 (1) 설치, (2) 첫 GRAPE 펄스 최적화,
(3) 파일 내보내기, (4) 예제 실행까지 마친 것입니다. 축하합니다!
