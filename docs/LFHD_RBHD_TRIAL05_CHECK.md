# Trial 05: Facing 회전 및 보행 방향 확인 결과

## 1. Facing 회전 사용 여부

- **CLI**: `--static-facing`, `--dynamic-facing`은 **기본값 `None`** (`src/marker_label/cli.py` 41–54).
- **파이프라인**: 회전은 **둘 다 설정된 경우에만** 적용됨 (`pipeline.py` 201–204):
  ```python
  if static_facing_axis and dynamic_facing_axis:
      R = rotation_for_facing_axes(...)
      points_d_body_for_matching = points_d_body @ R.T
  ```
- **예시 명령** (대화 요약):  
  `python -m marker_label.cli 'data/BBA01 Cal 01.c3d' 'data/BBA01 Trial 05.c3d' --z-band-by-rank --left-side-positive-y -o out/BBA01_trial05`  
  → **`--static-facing` / `--dynamic-facing` 없음.**

**결론:** 위 예시대로만 돌렸다면 **Facing 회전은 사용되지 않음**.  
다른 스크립트나 명령으로 실행했다면, 그 호출에 `--static-facing`과 `--dynamic-facing`이 함께 들어갔는지 확인하면 됨.

---

## 2. 보행 방향 (Trial 05)

`scripts/report_lr_ap_axes.py "data/BBA01 Trial 05.c3d"` 실행 결과:

| 항목 | 값 |
|------|-----|
| Trial | data/BBA01 Trial 05.c3d |
| Frames / Points | 1072 / 44 |
| Centroid X | first = -1843.0 mm, last = 4568.1 mm |
| mean dX | **5.99 mm/frame** |
| **Walking direction (pipeline)** | **+X (subject walks toward increasing X)** |
| d_back | [-1.0, 0.0] → posterior = -X |
| A/P | **anterior = larger X**, posterior = smaller X |

**결론:** Trial 05에서는 **보행 방향이 +X로 정상**이며, `d_back`에 따른 A/P 해석도 “큰 X = 전방”으로 맞음.  
즉, **“보행 방향 잘못됨 → d_back 반대 → LFHD/RBHD 스왑”** 가능성은 이 트라이얼에서는 낮음.

---

## 3. LFHD/RBHD 스왑 원인 정리 (Trial 05 기준)

- **Facing 회전**: 예시 명령 기준으로는 **미사용** → 좌표계 불일치(문서 §2)로 설명하기 어려움.
- **보행 방향**: **+X로 정상** → 잘못된 `wdx`/`d_back`으로 설명하기 어려움.

가능한 다른 원인:

1. **실제 실행에 facing 옵션 사용**  
   다른 명령/스크립트에서 `--static-facing` / `--dynamic-facing`을 넣었다면, 그때는 회전으로 인한 A/P 불일치 가능.
2. **헤드 기하로 인한 A/P cross-split**  
   X만으로 전/후를 나누면, 헤드 각도나 마커 위치 때문에 “전방좌·후방우” 한 쌍이 반대 그룹에 들어가 LFHD/RBHD만 스왑되는 경우 (문서 “Why only LFHD and RBHD swap” 참고).
3. **Best frame / 포인트 순서**  
   best frame 선택이나 해당 프레임의 4점 순서에 따라 동일 로직이라도 다른 할당이 나올 수 있음.

**권장:**  
- 당장 고치려면: `--head-swap-lfhd-rbhd` 로 LFHD↔RBHD만 스왑.  
- 전방/후방이 반대로 잡혔다고 보이면: `--head-anterior-larger-x` 또는 `--head-anterior-smaller-x` 로 헤드 A/P만 덮어쓰기.
