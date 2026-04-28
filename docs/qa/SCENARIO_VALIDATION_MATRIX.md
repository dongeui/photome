# Scenario Validation Matrix

## Stage 1 — Caching Pipeline (T1~T15)

| Scenario | Trigger | Expected User-visible Result | Expected Internal State | Priority | Verified |
|---|---|---|---|---|---|
| new image file | NAS scan finds unseen image | file appears in `/media` after processing | `discovered -> metadata_done -> thumb_done` | Critical | No |
| new video file | NAS scan finds unseen video | file appears with duration and thumb | keyframe or poster asset registered | Critical | No |
| moved file | same fingerprint, different path | media record stays same id | `current_path` updated, `file_id` unchanged | Critical | No |
| deleted file | previous file missing from NAS | item not broken, marked unavailable | status becomes `missing` | Critical | No |
| partial upload | file size changes between checks | item not processed too early | stays `waiting_stable` | Critical | No |
| NAS offline | mount unavailable during polling | API still serves existing metadata | scan run fails gracefully, no data corruption | Critical | No |
| image metadata missing | EXIF absent or malformed | item still listed | metadata fallback used, no crash | High | No |
| ffprobe failure | broken or unsupported video | item still visible with error note | stage moves to `error` or partial metadata | High | No |
| thumb generation failure | Pillow/OpenCV error | item detail still resolves | error logged, retryable state saved | High | No |
| filter by media type | `GET /media?media_type=image` | only requested type returned | query uses indexed filter | Medium | No |

## Stage 2 — Semantic Enrichment and NL Search (T16~T24)

| Scenario | Trigger | Expected User-visible Result | Expected Internal State | Priority | Verified |
|---|---|---|---|---|---|
| GPS → 지명 태그 | 좌표 포함 이미지 처리 | `/media/filter?tag=제주` 로 검색 가능 | `place:country/region/city` 계층 태그 생성, 역지오코딩 캐시 적중 | Critical | No |
| CLIP 임베딩 생성 | 신규 이미지 썸네일 완료 후 | 해당 파일이 자연어 검색 후보로 등장 | `DerivedAsset(asset_kind="embedding_clip")` 등록, 상태 `embedding_done` | Critical | No |
| 자연어 쿼리 파싱 | `/search?q=작년에 제주에서 가족이랑` | 상위 결과가 조건에 부합 | 파서 JSON `{time, place, people, semantic}` 출력, 필드별 필터 적용 | Critical | No |
| 파서 실패 폴백 | LLM 응답이 JSON 스키마 위반 | 쿼리는 여전히 결과 반환 | `semantic-only` 폴백 동작, 메타 필터 생략 | High | No |
| 사람 라벨 갱신 | PATCH `/people/{id}` 로 `엄마`로 변경 | 이후 `엄마` 쿼리에서 그 사람 사진 반환 | `Person.display_name` 갱신, 관련 `tag_type="person"` 일괄 리네임 | Critical | No |
| 사람 그룹("가족") | `person_group` 태그 지정 | "가족" 쿼리가 지정된 사람들 사진을 반환 | `tag_type="person_group"` 다건 매핑 | High | No |
| 빈 결과 쿼리 | 일치 없음 | 빈 리스트 + 200 OK | 500 금지, 로그에 쿼리 구조 저장 | Medium | No |
| 시간 표현 정규화 | `작년`, `지난달`, `2024년 여름` | 쿼리 시점 기준으로 date range 환산 | 파서가 절대 범위로 변환 후 메타 필터 적용 | High | No |
| 임베딩 모델 버전 업그레이드 | `embedding_clip` asset_version 상승 | 기존 데이터 검색 품질 유지 | 야간 배치 또는 스크립트로 재인덱싱, 구버전 임베딩은 즉시 삭제되지 않음 | Medium | No |
| OCR 텍스트 검색(선택) | 메뉴판 이미지 처리 | OCR 텍스트로 검색 가능 | `tag_type="ocr_text"` 저장, 한글 인식 성공 | Low | No |

