# Phase 2 Search Technical Review
updated 2026-04-29

이 문서는 Phase 2의 종착지 기술 방향만 남긴다. 목표는 Google Photos / Lens / 검색 엔진급의 전면 성능이 아니라, 로컬-first 개인 사진 서버에서 유사한 검색 경험을 최대한 재현하는 것이다.

## 목표 경험

사용자는 파일명이나 폴더를 기억하지 않아도 아래 방식으로 사진을 찾을 수 있어야 한다.

- 키워드: `영수증`, `스크린샷`, `강아지`, `제주`
- 자연어: `작년 여름 바다에서 가족이랑 찍은 사진`
- 복합 조건: `지난달 카톡 오류 캡처`, `엄마랑 카페에서 찍은 사진`
- 개념 검색: `생일 케이크`, `여행 느낌`, `문서 같은 사진`
- 사람/장소/시간: `아빠 제주 2024`, `아이 졸업식`
- 텍스트 포함 이미지: OCR로 읽힌 메뉴, 영수증, 화면 문구

## 외부 제품에서 가져올 원칙

- Google Photos는 검색 축을 `사람/장소/사물/문서/앨범`처럼 사용자가 이해하기 쉬운 facet으로 노출한다.
- Ask Photos류 자연어 검색은 단순 벡터 검색이 아니라 사진 내용을 이해하고 이미지 속 텍스트까지 활용한다.
- Lens/AI Mode류 검색은 이미지나 텍스트 입력을 받아 여러 하위 검색으로 분해하고 결과를 종합한다.
- 대규모 검색 서비스는 dense semantic search와 exact keyword search를 함께 쓰고, 이후 reranking으로 결과를 정리한다.

Photome의 대응 원칙:

- `search_documents`를 canonical semantic profile로 둔다.
- keyword/OCR/person/place/date/visual channel을 분리한다.
- 각 channel 후보를 독립적으로 가져온 뒤 RRF와 boost/rerank로 결합한다.
- 모델 호출은 Phase 2 cycle에서만 수행하고, request path에서는 가능한 색인 조회만 한다.
- 전량 재처리 대신 missing/stale/version mismatch만 처리한다.

## 현재 확정된 기반

- `SearchDocument`: filename, relative path, annotation, OCR, tags, people, places, analysis signals, embedding refs를 한 row로 집계한다.
- `search_documents_fts`: SQLite FTS5 keyword acceleration layer다.
- `QueryPlanner`: keyword/OCR/person/place/date/visual intent를 rule-based로 분해한다.
- `VectorIndexBackend`: vector search adapter interface다.
- `LocalNumpyVectorIndex`: 현재 기본 exact vector backend다.
- `run_semantic_maintenance()`: Phase 2 cycle에서 missing/stale/version mismatch media만 처리한다.
- `semantic_search_version`: search document rebuild 기준이다.

## 종착지 아키텍처

```text
Phase 1
  scanner -> fingerprint -> metadata -> thumbnail/keyframe -> media status

Phase 2 cycle
  processed media only
    -> OCR
    -> image signals
    -> visual embedding
    -> face/person/place signals
    -> caption/scene summary
    -> search_documents
    -> FTS index
    -> vector index

Request-time search
  query
    -> QueryPlanner
    -> keyword / OCR / tag / person / place / date / vector channels
    -> fusion
    -> rerank
    -> explanations + filters
```

## Image Understanding Stack

### 1. Fast deterministic signals

항상 먼저 처리한다.

- filename/path token
- EXIF datetime
- GPS coordinate and coarse place
- image dimensions/aspect ratio
- screenshot/document/receipt-like signals
- brightness/edge density/text density

목적:

- 모델 없이도 기본 검색 가능
- 모델 실패 시 fallback
- query planner와 ranking의 cheap feature

### 2. OCR

현재 Tesseract 기반 OCR은 유지하되, 한국어/스크린샷/문서 품질을 위해 PaddleOCR adapter를 추가한다.

우선순위:

1. Tesseract 유지: 가벼운 기본 OCR
2. PaddleOCR optional provider: Korean, screenshot, receipt, document 품질 개선
3. OCR confidence와 block 위치를 `search_documents`에 요약
4. OCR 결과는 FTS와 OCR channel 모두에 공급

### 3. Visual Embedding

현재 OpenCLIP path를 유지하되 provider를 분리한다.

provider 후보:

- OpenCLIP: 로컬-first, OSS, 설치 가능성 높음
- SigLIP 계열: text-image retrieval 품질 후보
- external multimodal embedding: 비용/프라이버시 허용 시 optional

규칙:

- request-time에 이미지 embedding 생성 금지
- Phase 2 cycle에서 versioned embedding 생성
- `VectorIndexBackend`로 검색 엔진 교체 가능하게 유지

### 4. Caption / Scene Summary

caption은 검색 품질을 크게 올리지만 비용이 크므로 optional provider로 둔다.

caption output은 아래처럼 분리한다.

- `short_caption`: 한 줄 장면 설명
- `objects`: 사물
- `activities`: 행동/이벤트
- `setting`: indoor/outdoor/place 느낌
- `people_description`: 사람 수/구도, 신원 추론 금지
- `text_summary`: OCR 요약

caption은 `search_documents.semantic_text`와 FTS에 들어간다.

### 5. People / Place

사람:

- face cluster는 자동 person id를 만든다.
- 사용자가 display name과 group을 붙인다.
- query planner는 `엄마`, `가족`, `친구`를 person/person_group channel로 보낸다.

장소:

- GPS coarse tag는 이미 가능하다.
- reverse geocoding provider/cache를 붙여 country/region/city/place hierarchy를 만든다.
- 장소명은 keyword/FTS/tag/place filter 모두에서 검색 가능해야 한다.

## Query Stack

### QueryPlanner

현재 rule-based를 canonical fallback으로 유지한다.

output:

- `keyword_query`
- `visual_queries`
- `date_from`, `date_to`
- `person_terms`
- `place_terms`
- `ocr_terms`
- `visual_terms`
- `intent`

추후 LLM planner는 optional provider다. LLM 결과는 반드시 schema validation을 통과해야 하며 실패 시 rule-based planner로 fallback한다.

### Retrieval Channels

각 channel은 독립적으로 top-K 후보를 반환한다.

- FTS channel: `search_documents_fts`
- OCR channel: `media_ocr`, OCR grams
- tag channel: auto/custom/person/place/group tags
- person channel: person id/group id
- place channel: GPS/reverse geocoding tags
- date channel: metadata date range filter
- vector channel: visual embedding similarity
- caption channel: caption text FTS/vector

### Fusion / Reranking

1차:

- RRF로 channel 후보 병합
- exact OCR/tag/person/place boost
- date/place/person hard filter 적용

2차:

- query intent별 가중치 조정
- screenshot/document query는 OCR/FTS 우선
- people query는 person/face 우선
- broad visual query는 vector/tag 우선
- 자연어 복합 query는 date/person/place filter + vector/FTS 혼합

3차 optional:

- cross-encoder/reranker adapter
- LLM explanation only, ranking 결정권은 제한

## Storage / Index Strategy

현재:

- SQLite metadata
- `search_documents`
- SQLite FTS5
- local NumPy exact vector

다음 단계:

- `VectorIndexBackend` adapter 추가
  - local NumPy: baseline
  - FAISS: local ANN, dependency 추가 적음
  - LanceDB: embedded multimodal table + hybrid search
  - Qdrant: separate service, large-scale/hybrid/filtering

선택 기준:

- 1만 장 이하: SQLite FTS5 + local NumPy로 충분
- 1만~10만 장: FAISS 또는 LanceDB 검토
- 10만 장 이상 / 다중 클라이언트 / 고급 filtering: Qdrant 검토

## 품질 평가 기준

실제 이미지가 충분하지 않을 때도 synthetic QA를 먼저 만든다.

- planner unit tests: 쿼리 intent 분해 정확도
- search document tests: semantic profile materialization
- FTS tests: keyword/OCR/tag recall
- vector backend contract tests: adapter 교체 가능성
- fusion tests: channel agreement ranking
- regression queries:
  - `작년 여름 바다에서 가족이랑 찍은 사진`
  - `지난달 카톡 오류 캡처`
  - `엄마랑 카페`
  - `영수증`
  - `생일 케이크`
  - `아기 사진`
  - `제주 여행`

실제 이미지 확보 후 평가:

- recall@20
- top-5 useful rate
- no-result false negative rate
- OCR hit rate
- duplicate/near-duplicate grouping quality
- query latency p50/p95

## Roadmap

### R1 — 검색 구조 안정화

- [x] `search_documents`
- [x] FTS5 index
- [x] Phase 2 maintenance cycle
- [x] QueryPlanner
- [x] VectorIndexBackend
- [x] `/search/debug` and dashboard search tuning panel
- [ ] synthetic NL QA 확장

### R2 — 이미지 이해 품질 개선

- [ ] PaddleOCR provider
- [ ] caption provider interface
- [ ] person group API
- [ ] reverse geocoding provider/cache
- [ ] richer auto tags from image signals and captions

### R3 — 검색 랭킹 고도화

- [ ] intent-specific channel weights
- [ ] hard/soft filter distinction
- [ ] reranker interface
- [ ] result explanations
- [ ] user feedback signal: hide/promote/correct tag

### R4 — scale path

- [ ] FAISS adapter benchmark
- [ ] LanceDB adapter benchmark
- [ ] Qdrant adapter benchmark
- [ ] index rebuild CLI
- [ ] index health/status page

## 결론

Photome Phase 2의 경쟁력은 단일 모델 성능이 아니라 개인 사진 한 장을 검색 가능한 semantic profile로 안정적으로 만들고, 쿼리를 여러 retrieval channel로 분해해 재현성 있게 결합하는 데 있다. 로컬-first 제약에서는 `search_documents + FTS5 + vector adapter + rule-based planner`가 가장 안전한 중심축이다.
