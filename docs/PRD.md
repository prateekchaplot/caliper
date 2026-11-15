# Product Requirements Document (PRD)

**Project Name:** Docs Assistant with Measured Quality (Project 1)

## 1. Product Overview
<!-- What the product is, what it enables users to accomplish, its primary purpose, and its overall scope. Keep this section concise. -->

- A question-answering assistant over one public product documentation set. A user asks a natural-language question and receives an answer they can verify: the answer cites the exact source page and quotes the supporting passage, and questions the documentation cannot support are refused instead of guessed.
- The product has a second, equally important purpose: proof of quality. Every change to the assistant is measured on the same frozen 60-question set, and the results of each version are published side by side so improvement is demonstrated, not claimed.
- Scope: a single-user tool that operates locally over exactly one documentation set. It makes no claim of coverage beyond that corpus.

**Assumptions**

- One public documentation set is selected and frozen before the first measured baseline run. The leading candidate is the Azure Functions and Azure Service Bus documentation (a few hundred pages); the final selection is a week-1 decision recorded in the project log.
- The answering component and the content-representation approach are selected once, in week 1, and held fixed across all evaluated versions, so version-to-version differences isolate retrieval behavior.
- The 60-question golden set is authored by the project owner before any tuning begins and is never edited during a measured run.
- No performance or accuracy targets exist today; all baselines are established during the first measured (baseline) run.
- The project is delivered by December 13, 2026, at approximately 10 hours of work per week.
- The source documentation's licensing permits linking to its pages and storing derived test data; the corpus itself is not redistributed.

## 2. Problem Statement
<!-- The business problem, current user difficulties, existing workflow limitations, and why a dedicated solution is needed. Focus on the problem, not the proposed implementation. -->

- Most AI question-answering demos answer confidently but cannot show that their answers are correct. There is no citation to check and no measure of how often the answer is right.
- When such tools are applied to messy, real-world documentation, their quality is unknown: nobody measures how often the correct passage is retrieved, how often a citation actually supports the claim, or how often an unanswerable question is refused rather than hallucinated.
- Without measurement, improvement cannot be demonstrated. A change to how content is retrieved and ranked "feels" better or worse, but no evidence backs it — which is exactly the gap a technical reviewer probes first.
- The user who needs a trustworthy answer has no way to verify where an answer came from, and the assistant has no principled way to say "this is not in the docs" instead of guessing.
- A dedicated solution is needed that treats verifiable answers and measured quality as first-class product behavior: cite sources, refuse unsupported questions, and prove each improvement with a repeatable evaluation on an unchanged question set.

## 3. Product Goals
<!-- A numbered or bulleted list of measurable or outcome-oriented goals. Each goal should describe a business or user outcome. -->

- **G1 — Verifiable answers:** every answer cites the exact source page and quotes the supporting passage, so a user can confirm the claim in one step.
- **G2 — Honest refusal:** when the retrieved documentation does not support an answer, the assistant explicitly responds that the answer is not found in the docs, rather than guessing.
- **G3 — Demonstrated improvement:** three progressively improved versions are measured on the identical frozen 60-question set, and the comparison table is published so improvement is shown, not claimed.
- **G4 — Effortless adoption:** a person with no prior involvement can obtain the project and run it end to end using the setup documentation alone.

## 4. Target Users
<!-- Primary users, secondary users, their responsibilities, and their relationship with the product. Do not invent unnecessary user personas. -->

- **Primary — Support engineer or developer:** asks natural-language questions about the covered product's documentation and needs an answer they can independently verify through citations. Interacts with the assistant through the question interface and the review screen (when delivered).
- **Secondary — Technical reviewer (interviewer, mentor, or collaborator):** obtains the project, starts it using the setup documentation alone, and inspects the evaluation report to judge whether the claimed quality improvements are real. Relates to the product as an evaluator of both the assistant and its evidence.
- **Operational — Project maintainer (the builder):** ingests the documentation set, curates and freezes the golden question set, runs evaluations between versions, and inspects logged traces to debug and improve retrieval.

## 5. Functional Requirements
<!-- Requirements with unique identifiers in the format "FR-01: Requirement Name". For each requirement: describe the expected product behavior, the user's interaction with the system, relevant validation and failure scenarios, and authorization or approval requirements where applicable. Use concise bullet points beneath each requirement. Order requirements according to the primary user workflow. -->

Requirements FR-01 through FR-11 deliver goals G1–G4 and are required for the MVP. FR-12 through FR-14 are stretch requirements, cut first if the schedule slips.

### FR-01: Documentation Ingestion
- **Priority:** Must
- The system must ingest a chosen public documentation set published as Markdown or HTML pages.
- For every ingested page, the system must preserve the page URL and the heading path so that any stored piece of content can be traced back to its exact location in the source documentation.
- When a page is malformed or cannot be fully processed, the system must skip it and report it in an ingestion summary instead of failing the entire ingestion run.

### FR-02: Repeatable Content Indexing
- **Priority:** Must
- The system must divide ingested content into independently retrievable units and make them searchable.
- Re-running ingestion on the same documentation set must be repeatable: it must produce a consistent, non-duplicated searchable index without manual cleanup.
- The system must support at least two distinct content-segmentation configurations, so that a change in segmentation can be evaluated as a separate version (see FR-10).

### FR-03: Question Submission and Answer Generation
- **Priority:** Must
- The user submits a single natural-language question through a request–response interface. Each question stands alone; the assistant holds no conversation memory across sessions.
- The system must generate its answer using only the documentation passages retrieved for that question.
- The response must contain the answer (or a refusal, see FR-05), the citations (see FR-04), and a confidence flag indicating how strongly the retrieved passages supported the answer.
- Questions unrelated to, or outside the scope of, the ingested documentation must be handled by the refusal behavior of FR-05, not by an attempted answer.

### FR-04: Source Citations
- **Priority:** Must
- Every non-refusal answer must cite the source page (URL) and include a verbatim quoted passage from that page that supports the answer.
- Citations must be human-verifiable: a user must be able to open the cited page and locate the quoted passage on it.
- When an answer is supported by more than one passage (for example, an answer spanning two pages), each supporting passage must be cited.

### FR-05: Refusal of Unsupported Questions
- **Priority:** Must
- When the retrieved passages do not support an answer, the system must return an explicit "not found in the docs" response instead of an attempted answer.
- A refusal must not present speculative content as if it were an answer drawn from the documentation.
- The system must distinguish supported from unsupported questions in both directions — refusing unanswerable questions and not refusing answerable ones — with the accuracy of this distinction measured and reported (see FR-09 and Success Metrics).

### FR-06: Relevant Passage Ranking
- **Priority:** Must
- The system must rank retrieved content so that the most relevant passages for the question form the final, bounded set used to generate the answer.
- The final set supplied for answer generation must be small and fixed in size (five passages in the evaluated configuration).
- Ranking behavior must be observable through the evaluation harness so that ranking changes across versions can be compared (see FR-09 and FR-10).

### FR-07: Request Logging
- **Priority:** Must
- The system must log every question it receives, recording at minimum: the question text, the identifiers of the passages retrieved, the returned answer or refusal, the response latency, and the estimated processing cost.
- Logs must be retained so that any past request can be reconstructed and inspected for debugging and evaluation.

### FR-08: Frozen Golden Question Set
- **Priority:** Must
- The project must maintain a fixed set of 60 questions, authored in the style a support engineer would ask, before any tuning begins:
  - 40 answerable questions, each annotated with the documentation page that should be cited.
  - 10 questions whose answer spans two pages.
  - 10 unanswerable questions the documentation does not cover, which should be refused.
- The set must be frozen before the first measured run and must not be edited during or between measured runs.

### FR-09: Evaluation Harness
- **Priority:** Must
- The system must provide an evaluation capability that runs the full golden question set against any version of the assistant and records the outcome as a new, separately identified run.
- Evaluation runs must be append-only: a recorded run is never overwritten or altered by a later run.
- Each run must produce a results table covering, at minimum: retrieval hit rate within the final passage set, answer correctness on a manually spot-checked sample, citation accuracy, refusal accuracy, and per-question cost and latency.
- Any change to content segmentation, answer instructions, or the fixed answering/retrieval components must start a new logged run.

### FR-10: Version Comparison
- **Priority:** Must
- The project must deliver three evaluated versions on the identical frozen question set:
  - V1: baseline configuration.
  - V2: same retrieval approach as V1 with improved content segmentation.
  - V3: V2's segmentation with improved retrieval and passage ranking.
- The answering component and the content-representation approach must remain fixed across all three versions, so that measured differences are attributable to retrieval behavior.
- All three runs must be presented together in a single comparison table in the project's primary documentation.

### FR-11: Reproducible One-Step Setup
- **Priority:** Must
- A new user must be able to obtain the project, complete configuration from a provided sample configuration template, and start the full system with a single documented command.
- After setup, submitting a question must work end to end without additional manual steps.

### FR-12: Review Screen
- **Priority:** Should
- The system must provide a review screen that displays, for a given question: the question, the answer, the cited sources, and controls for the user to mark the answer as helpful or not helpful.
- Readability is the only presentation requirement; no styling effort beyond legibility is in scope.

### FR-13: Feedback Capture and Export
- **Priority:** Should
- Feedback submitted on the review screen must be stored and must be exportable in a form suitable for reviewing as candidate additions to the golden question set.

### FR-14: Per-Run Cost and Latency Summary
- **Priority:** Should
- Each evaluation run must produce a summary of the median and 95th-percentile latency and processing cost per question.

## 6. Non-Functional Requirements
<!-- Requirements with unique identifiers in the format "NFR-01: Requirement Name". Consider the applicable categories: reliability, security and access control, performance, usability, observability and auditability, maintainability, data integrity, availability. Only include categories relevant to the product. Describe expected quality attributes only. -->

- **NFR-01: Traceability and auditability** — every answer or refusal must be fully reconstructable after the fact: the question, the passages retrieved, the response returned, and its cost and latency must all be recoverable from logs.
- **NFR-02: Evaluation integrity** — golden question sets and recorded evaluation runs must be immutable once frozen; no mechanism may silently modify prior results.
- **NFR-03: Performance measurement** — the system must report median and 95th-percentile latency and processing cost per question for every evaluation run. No latency or cost targets are asserted before the baseline run; targets are set from measured baselines.
- **NFR-04: Verifiability and usability** — a citation must be checkable by a user in one step (open the cited page and find the quoted passage), and setup must require no expertise beyond following the provided documentation.
- **NFR-05: Graceful failure** — when any step of the question flow yields nothing usable (for example, retrieval returns no relevant passages), the system must respond with a clear refusal or error message rather than crashing or producing an ungrounded answer.
- **NFR-06: Licensing compliance** — the product must operate within the source documentation's license terms: it must link to source pages and store only its own project files and derived data, never redistributing the corpus itself.
- **NFR-07: Access model** — the tool is single-user and local; no multi-tenant access, accounts, or authentication is required.

## 7. MVP Scope
<!-- Divide into "In scope" (essential capabilities required for the initial usable product) and "Out of scope" (advanced capabilities, non-essential integrations, and future enhancements). Ensure the scope is realistic for an initial implementation. -->

**In scope**

- Documentation ingestion with page URL and heading-path preservation, and repeatable indexing (FR-01, FR-02).
- Question interface returning an answer, citations, and a confidence flag (FR-03, FR-04).
- Explicit refusal of unsupported questions (FR-05).
- Bounded, ranked final passage set (FR-06).
- Full request logging (FR-07).
- Frozen 60-question golden set and append-only evaluation harness (FR-08, FR-09).
- Three evaluated versions presented in one comparison table (FR-10).
- One-step reproducible setup from documentation alone (FR-11).
- Stretch, included only if the schedule allows and cut first if it does not: review screen with feedback capture, feedback export, and per-run cost/latency summary (FR-12, FR-13, FR-14).

**Out of scope**

- Model fine-tuning or training any component of the system.
- Multi-agent orchestration or complex agent workflows.
- User accounts, authentication, multi-tenancy, or hosted production deployment.
- Conversation memory across sessions; every question stands alone.
- Any claim of coverage beyond the single chosen documentation set.
- Extensive user-interface styling beyond readability.
- Automated continuous evaluation in a delivery pipeline; evaluation runs happen at version milestones.
- Automated answer judging as the sole arbiter of correctness; a manually spot-checked sample governs, with any judge disagreement reported.

## 8. Acceptance Criteria
<!-- Numbered, objectively verifiable conditions that determine whether the MVP is complete. Cover all critical functional requirements, important failure scenarios, and security and authorization checks where applicable. Describe observable outcomes; avoid implementation-specific instructions. -->

1. After ingestion, a question about content covered by a known documentation page returns an answer citing that page's URL and a quoted passage that appears on that page.
2. For an answer supported by two pages, the response cites both supporting passages.
3. When the retrieved passages do not support an answer, the response is an explicit "not found in the docs" refusal containing no speculative answer content.
4. The frozen 60-question set exists with its stated composition (40 single-page answerable, 10 two-page answerable, 10 unanswerable), is documented, and is shown to be unchanged across all measured runs.
5. Running the evaluation harness on the golden set produces a results table reporting retrieval hit rate, answer correctness on a spot-checked sample, citation accuracy, refusal accuracy, and per-question cost and latency for each evaluated version.
6. Three evaluation runs (V1, V2, V3) exist on the identical frozen question set, are presented together in one comparison table in the project's primary documentation, and no earlier run is overwritten.
7. Every recorded request can be reconstructed from logs, including the question, retrieved passage identifiers, the answer or refusal, latency, and estimated processing cost.
8. Running ingestion twice on the same documentation set produces a consistent, non-duplicated searchable index, with any skipped pages reported in an ingestion summary.
9. A person with no prior involvement can, using only the project's setup documentation and the sample configuration template, start the system with a single command and obtain an answer to a test question end to end.
10. Each stretch requirement (FR-12, FR-13, FR-14) is either delivered or explicitly recorded as cut, so the final scope is unambiguous.
11. The project demonstration explains one failure observed during evaluation and the specific change made to fix it.

## 9. Success Metrics
<!-- Measurable indicators of product success. For each metric: explain what it measures, how it relates to product goals, and suggest a baseline or target only when supported by available information. If no baseline exists, explicitly state that it must be established during initial testing. -->

No baselines exist today. Every metric below must be established during the initial (V1 baseline) measured run, and improvement targets are set from those baselines — not asserted beforehand.

- **Retrieval hit rate (final passage set):** the proportion of answerable golden questions for which a correct source page appears in the final ranked passages. Measures whether the assistant can find the right material at all. Directly supports G1 and G3.
- **Answer correctness:** the proportion of correct answers in a manual spot-check of 20 answers against the documentation. Measures end-to-end answer quality. Supports G1 and G3.
- **Citation accuracy:** the proportion of cited passages that actually support the answer's claim. Measures whether citations are trustworthy rather than decorative. Supports G1 and G3.
- **Refusal accuracy:** the proportion of unanswerable questions refused, combined with the proportion of answerable questions not refused. Measures honest behavior in both directions. Supports G2 and G3.
- **Latency and processing cost:** the median and 95th-percentile latency and estimated processing cost per question. Measures whether quality improvements remain practical. Supports G3 and constraints the iteration budget.
- **Version-over-version delta:** the measured change in the metrics above from V1 to V2 to V3. This is the headline success indicator for the product: a demonstrable improvement, shown in the comparison table. Supports G3.
- **Unaided setup success:** at least one person with no prior involvement starts the system and asks a question successfully using the setup documentation alone, verified during the final week. Supports G4.

## 10. Deliverables
<!-- Expected product and project deliverables, such as: working application, user interface, functional documentation, test cases and test results, evaluation report (if applicable), user documentation, demonstration, known limitations and future enhancements. -->

- Working question-answering application implementing FR-01 through FR-11, including the citation and refusal behavior.
- The frozen 60-question golden set, with documented construction rules and its frozen state recorded.
- Evaluation report: separately logged runs for V1, V2, and V3, and a single comparison table published in the project's primary documentation.
- Complete request-trace logs covering all measured runs.
- Reproducible setup package: setup documentation and a sample configuration template enabling one-command startup.
- Review screen with feedback capture and feedback export (if the stretch requirements FR-12 and FR-13 are delivered; otherwise a dated record that they were cut).
- Test cases and test results mapped to the acceptance criteria in Section 8.
- A short demonstration (approximately three minutes) explaining one observed failure and the fix applied.
- A documented list of known limitations and future enhancements.
