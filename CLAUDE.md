# Claude Code 협업 규칙

## 응답 방식

- 코드 수정에 대한 결과와 코드 변경 요약은 보여줄 필요 없다.
- 완료 후 핵심 변경점과 다음 액션만 간략히 전달한다.
- 막히지 않는 한 중간에 확인하지 않고 끝까지 진행한다.
- 작업 진행 중 터미널 출력이나 테스트 로그는 굳이 사용자에게 리포트하지 않는다. 컨텍스트 절약을 우선한다.
- 작업 단위가 끝나면 커밋하고 푸쉬까지 시도한다. 단, GitHub 인증 등 외부 권한 문제로 푸쉬가 막히면 최종에 짧게만 알린다.

## 코드 방침

- 하드코딩 금지: 한글 태그, 고유명사, 도메인 사전 등은 절대 코드에 박지 않는다.
  - 대신 DB 기반(TagVocabularyCache), 형태소 분석(KoNLPy), 또는 설정 파일로 해결한다.
- 특정 단어·목록이 필요하면 "왜 동적으로 못 하는가"를 먼저 검토한다.
- 범용·공통 개선에 집중하고, 케이스별 패치는 지양한다.

## 개선 계획 방식

- 새로운 개선 방향을 잡을 때는 스스로 3번 질문하고 리스트업한 뒤 진행한다.
- 각 질문은 "지금 코드가 X 상황에서 어떻게 동작하는가?"처럼 구체적인 failure mode를 짚는다.

## 프로젝트 특수 사항

- 한국어 검색: `app/services/search/tokenizer.py`의 `korean_nouns()` 경유, KoNLPy 없으면 heuristic fallback.
- 태그 어휘: `TagVocabularyCache` (5분 TTL, classmethod `invalidate()`)로 DB 태그를 검색 플래너에 주입한다.
- 하이브리드 검색 채널: OCR · CLIP(vector) · Shadow(FTS/태그) 3채널 RRF 결합.
- 검색 문서 버전: `semantic_search_version` 상수 + content hash 비교로 불필요한 FTS 재작성을 방지한다.
